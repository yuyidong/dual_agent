"""Paired phase-three experiment with baseline-inclusive validation selection."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from phase3_cross_evaluation import (
    build_models, cpu_state_dict, evaluate_pair, load_module_checkpoint,
    evaluate_forecaster_output_mmd,
    load_config, ScenarioDataset, make_train_test_loaders, make_complete_graph,
    DualAgentSystem, TorchLinDistFlowEvaluator, evaluate_forecaster_loss,
    train_joint_epoch, train_surrogate_epoch, split_epochs, plot_cross_matrix,
)
from dual_agent.training.loops import disable_training_dropout


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/ieee13.yaml')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[7])
    p.add_argument('--forecast-epochs', type=int, default=300)
    p.add_argument('--surrogate-epochs', type=int, default=200)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--dropout', action='store_true')
    p.add_argument('--surrogate-lr', type=float, default=None)
    p.add_argument('--arms', nargs='+', choices=['forecast_only', 'forecast_budget', 'alternating'],
                   default=['forecast_only', 'alternating'])
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = load_config(args.config)
    dataset = ScenarioDataset('data/ieee13.npz')
    # Keep the original pretraining test split fixed across optimization seeds.
    original_train, test = make_train_test_loaders(
        dataset, batch_size=c.training.batch_size, test_fraction=.2, seed=c.seed)
    ids = list(original_train.dataset.indices)
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(20260922)).tolist()
    nval = max(1, len(ids)//5)
    val_ids = [ids[i] for i in perm[:nval]]
    train_ids = [ids[i] for i in perm[nval:]]
    val = DataLoader(Subset(dataset, val_ids), batch_size=c.training.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adjacency = make_complete_graph(dataset.pv_history.size(2)).to(device)
    evaluator = TorchLinDistFlowEvaluator(c.network.dss_master, c.network.pv_systems,
        c.network.batteries, c.lindistflow, c.problem.interval_hours).to(device)
    weights = [c.training.operating_cost_loss_weight, c.training.voltage_violation_loss_weight,
        c.training.line_flow_violation_loss_weight, c.training.kw_violation_loss_weight,
        c.training.soc_violation_loss_weight, c.training.terminal_soc_loss_weight]

    @torch.no_grad()
    def assess(f, s, loader):
        f.eval(); s.eval()
        totals = {}
        count = 0
        for batch in loader:
            b = {k:v.to(device) for k,v in batch.items()}
            z = f(b['pv_history'], adjacency)
            dispatch = s(z, b['load_forecast'], adjacency)['dispatch']
            result = evaluator(dispatch, b['pv_target'].unsqueeze(1), b['load_forecast'])
            n = len(b['pv_history']); count += n
            for k in ['operating_cost','voltage_violation','line_flow_violation',
                      'kw_violation','soc_violation','terminal_soc_deviation']:
                totals[k] = totals.get(k, 0.) + result[k].mean().item()*n
        totals = {k:v/count for k,v in totals.items()}
        totals['objective'] = sum(w*totals[k] for w,k in zip(weights, totals))
        return totals

    results = []
    for seed in args.seeds:
        for arm in args.arms:
            torch.manual_seed(seed); np.random.seed(seed)
            f,s = build_models(c, dataset.pv_history.size(2))
            load_module_checkpoint(f, Path('forecaster.pt'))
            load_module_checkpoint(s, Path('surrogate.pt'))
            system = DualAgentSystem(f,s).to(device)
            if not args.dropout:
                disable_training_dropout(system)
            fo = torch.optim.AdamW(f.parameters(), lr=c.training.joint_learning_rate)
            so = torch.optim.AdamW(s.parameters(), lr=(args.surrogate_lr
                if args.surrogate_lr is not None else c.training.joint_surrogate_learning_rate))
            train = DataLoader(Subset(dataset, train_ids), batch_size=c.training.batch_size,
                shuffle=True, generator=torch.Generator().manual_seed(seed))
            cap = evaluate_forecaster_loss(f, train, adjacency, device)*(1+c.training.joint_forecast_loss_tolerance)
            records = []
            states = [(cpu_state_dict(f), cpu_state_dict(s))]
            best_pair = states[0]
            best_score = assess(f,s,val)['objective']
            forecast_budget = args.forecast_epochs + (args.surrogate_epochs if arm == 'forecast_budget' else 0)
            for r,(nf,ns) in enumerate(zip(split_epochs(forecast_budget,args.rounds),
                                          split_epochs(args.surrogate_epochs,args.rounds))):
                for stage, epochs, model, optimizer in [('F',nf,f,fo),('S',ns,s,so)]:
                    if stage == 'S' and arm != 'alternating': continue
                    # Paired forecast shuffles do not depend on surrogate training length.
                    train.generator.manual_seed(seed*1000+r*2+(stage=='S'))
                    before = assess(f,s,val)
                    best = before
                    state = cpu_state_dict(model)
                    opt_state = copy.deepcopy(optimizer.state_dict())
                    selected_epoch = 0
                    for epoch in range(epochs):
                        if stage == 'F':
                            train_joint_epoch(system,evaluator,train,fo,adjacency,device,cap,
                                c.training.joint_forecast_constraint_weight,
                                c.training.joint_forecast_loss_weight,*weights,
                                max_grad_norm=c.training.max_grad_norm)
                        else:
                            train_surrogate_epoch(s,evaluator,train,so,adjacency,device,f,*weights,
                                max_grad_norm=c.training.max_grad_norm)
                        if (epoch+1)%args.interval == 0 or epoch+1 == epochs:
                            metrics = assess(f,s,val)
                            # Terminal SOC remains a soft objective term; report it separately.
                            feasible = all(metrics[k] <= before[k]+1e-6 for k in
                                ['voltage_violation','line_flow_violation','kw_violation',
                                 'soc_violation'])
                            if feasible and metrics['objective'] < best['objective'] and metrics['operating_cost'] < before['operating_cost']:
                                best = metrics; state = cpu_state_dict(model)
                                opt_state = copy.deepcopy(optimizer.state_dict()); selected_epoch = epoch+1
                    candidate_end = assess(f,s,val)
                    model.load_state_dict(state); optimizer.load_state_dict(opt_state)
                    records.append(dict(round=r+1,stage=stage,before=before,after=best,
                                        candidate_end=candidate_end,selected_epoch=selected_epoch))
                    print(seed,arm,r+1,stage,'selected',selected_epoch,
                          'cost',round(before['operating_cost'],3),round(best['operating_cost'],3),flush=True)
                states.append((cpu_state_dict(f),cpu_state_dict(s)))
                score = assess(f,s,val)['objective']
                if score < best_score:
                    best_score=score; best_pair=states[-1]
            f.load_state_dict(best_pair[0]); s.load_state_dict(best_pair[1])
            result = dict(seed=seed,arm=arm,validation=assess(f,s,val),test=assess(f,s,test),records=records)
            results.append(result)
            torch.save({'forecaster':best_pair[0],'surrogate':best_pair[1]},out/f'{arm}_{seed}.pt')
            if arm == 'alternating':
                matrix = np.empty((len(states),len(states)))
                for i,(_,ss) in enumerate(states):
                    s.load_state_dict(ss)
                    for j,(ff,_) in enumerate(states):
                        f.load_state_dict(ff); matrix[i,j]=assess(f,s,test)['operating_cost']
                np.save(out/f'matrix_{seed}.npy',matrix)
                forecaster_mmd = evaluate_forecaster_output_mmd(
                    f, [ff for ff,_ in states], test, adjacency, device,
                    reference="adjacent"
                )
                result['forecaster_output_mmd_adjacent'] = forecaster_mmd
                plot_cross_matrix(
                    matrix/matrix[0,0], out/f'matrix_{seed}.png', forecaster_mmd,
                    mmd_reference="adjacent"
                )
            (out/'results.json').write_text(json.dumps(dict(arguments=vars(args),
                config=Path(args.config).read_text(),train_indices=train_ids,validation_indices=val_ids,
                test_indices=list(test.dataset.indices),
                checkpoints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [Path('forecaster.pt'),Path('surrogate.pt')]},
                limitation='Validation was held out only for phase 3; pretrained checkpoints may have seen it.',
                results=results),indent=2))
            print('FINISHED',seed,arm,result['test'],flush=True)


if __name__ == '__main__':
    main()
