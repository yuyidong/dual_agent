"""Paired station perturbations with a multi-battery LinDistFlow MILP reference."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from dual_agent.config import load_config
from dual_agent.models.dual_agent import DualAgentSystem
from dual_agent.opendss.lindistflow_torch import TorchLinDistFlowEvaluator
from dual_agent.training.dataset import ScenarioDataset, make_complete_graph, make_train_validation_test_loaders, validate_dataset_matches_config

# Reuse architecture construction, not training or evaluation state.
sys.path.insert(0, str(ROOT / 'scripts/experiments/phase3_iterative_adaptation_cross_evaluation'))
from phase3_cross_evaluation import build_models


class MechanisticOPF:
    def __init__(self, evaluator, config):
        self.e = evaluator
        e = evaluator
        s, h = config.problem.scenarios, config.problem.horizon_steps
        b = e.battery_count
        self.s, self.h = s, h
        subtree = np.zeros((e.branch_count, e.bus_count))
        path = np.zeros((e.bus_count, e.branch_count))
        parents, children = e.parent_idx.numpy(), e.child_idx.numpy()
        for edge in reversed(e.edge_order.tolist()):
            child = children[edge]
            subtree[edge, child] = 1
            for downstream in np.flatnonzero(parents == child):
                subtree[edge] += subtree[downstream]
        for edge in e.edge_order.tolist():
            path[children[edge]] = path[parents[edge]]
            path[children[edge], edge] = 1
        self.subtree = subtree
        self.path = path
        self.net = cp.Parameter((s*h, e.branch_count))
        self.q = cp.Parameter((s*h, e.branch_count))
        dis, ch = cp.Variable((h,b), nonneg=True), cp.Variable((h,b), nonneg=True)
        mode = cp.Variable((h,b), boolean=True)
        self.dispatch = dis-ch
        rating = e.kw_rated.numpy()
        constraints = [dis <= cp.multiply(mode, rating), ch <= cp.multiply(1-mode, rating)]
        delta = cp.multiply(ch, e.interval_hours*e.charge_efficiency.numpy()/e.kwh_rated.numpy()) - cp.multiply(dis, e.interval_hours/(e.kwh_rated.numpy()*e.discharge_efficiency.numpy()))
        soc = e.soc_initial.numpy() + cp.cumsum(delta, axis=0)
        constraints += [soc >= e.soc_min.numpy(), soc <= e.soc_max.numpy()]
        flow = self.net - cp.vstack([self.dispatch]*s) @ subtree[:, e.battery_bus_idx.numpy()].T/1000
        voltage = 1-2*(cp.multiply(flow, e.branch_r.numpy()) + cp.multiply(self.q, e.branch_x.numpy())) @ path.T
        root = cp.sum(flow[:, e.root_edges.numpy()], axis=1)
        prices = np.tile(e._energy_price(h, torch.device('cpu'), torch.float64).numpy(), s)
        operating = 1000*e.interval_hours/s*(prices @ cp.pos(root) + e.curtailment_per_kwh*cp.sum(cp.pos(-root))) + e.degradation_per_kwh*e.interval_hours*cp.sum(dis+ch)
        voltage_penalty = cp.sum(cp.pos(e.voltage_lower_sq-voltage)+cp.pos(voltage-e.voltage_upper_sq))/(s*h*e.bus_count)
        line_penalty = cp.sum(cp.pos(cp.abs(flow)-e.branch_limit.numpy()))/(s*h*e.branch_count)
        terminal = cp.sum(cp.pos(cp.abs(soc[-1]-e.soc_initial.numpy())-e.terminal_soc_tolerance))/b
        w = config.training
        self.problem = cp.Problem(cp.Minimize(w.operating_cost_loss_weight*operating + w.voltage_violation_loss_weight*voltage_penalty + w.line_flow_violation_loss_weight*line_penalty + w.terminal_soc_loss_weight*terminal), constraints)
        self.weights = w

    def solve(self, scenarios, load):
        e = self.e
        with torch.no_grad():
            kw, kvar = e._expand_load(load.double())
        pv = scenarios.numpy().reshape(self.s*self.h, -1) @ e.pv_bus_matrix.numpy()
        self.net.value = (np.tile(kw[0].numpy(), (self.s,1))-pv) @ self.subtree.T/1000
        self.q.value = np.tile(kvar[0].numpy(), (self.s,1)) @ self.subtree.T/1000
        self.problem.solve(solver='HIGHS', highs_options={'mip_rel_gap':1e-7, 'time_limit':60.0, 'threads':1}, verbose=False)
        if self.problem.status != cp.OPTIMAL:
            raise RuntimeError(f'OPF did not converge: {self.problem.status}')
        dispatch = torch.as_tensor(self.dispatch.value.copy(), dtype=torch.float64).unsqueeze(0)
        with torch.no_grad():
            m = e(dispatch, scenarios.double(), load.double())
        w = self.weights
        objective = sum(getattr(w, key+'_loss_weight')*float(m[key].item()) for key in ['operating_cost','voltage_violation','line_flow_violation']) + w.terminal_soc_loss_weight*float(m['terminal_soc_deviation'].item())
        if not np.isclose(objective, self.problem.value, rtol=2e-5, atol=0.02):
            raise AssertionError(f'OPF/evaluator mismatch: {objective} vs {self.problem.value}')
        if float(m['soc_violation'].item()) > 1e-6 or float(m['kw_violation'].item()) > 1e-4:
            raise AssertionError('Infeasible battery dispatch')
        return dispatch


def perturb(base, capacity, mask, station, epsilon, direction):
    shifted = base.clone()
    shifted[..., station] = base[..., station] + direction*epsilon*capacity[station]*mask[..., station]
    return shifted


def common_perturbation_mask(base, capacity, daylight, epsilon, scale):
    # A shared support preserves identical error timing and symmetric amplitudes.
    margin = epsilon*scale
    feasible = ((base >= margin) & (base <= capacity-margin)).all(dim=-1, keepdim=True)
    active = daylight.bool().all(dim=-1, keepdim=True) & feasible
    return active.expand_as(base).to(base.dtype)


def station_ranks(values, rtol=1e-4):
    """Treat sub-0.01% differences as numerical ties, not station evidence."""
    values = np.asarray(values)
    order = np.argsort(-values,kind='stable')
    tolerance = max(float(np.abs(values).max())*rtol,1e-10)
    ranks = np.empty_like(values,dtype=float)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and abs(values[order[end]]-values[order[start]]) <= tolerance:
            end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    return ranks


def ranking_agreement(scores):
    ranks = np.array([station_ranks(v) for v in scores])
    if any(np.ptp(v) == 0 for v in ranks):
        return None
    return float(np.corrcoef(ranks)[0,1])


def paired_bootstrap(per_sample, seed=7, repeats=2000):
    values = np.asarray(per_sample)
    rng = np.random.default_rng(seed)
    means = np.stack([values[rng.integers(len(values), size=len(values))].mean(0) for _ in range(repeats)])
    maxima = means.max(-1, keepdims=True)
    normalized = np.divide(means, maxima, out=np.zeros_like(means), where=maxima > 1e-12)
    rhos = [r for v in means if (r := ranking_agreement(v)) is not None]
    return {
        'repeats':repeats, 'seed':seed, 'unit':'paired held-out sample',
        'normalized_ci95':np.quantile(normalized,[.025,.975],axis=0).tolist(),
        'importance_ci95':np.quantile(means,[.025,.975],axis=0).tolist(),
        'spearman_ci95':np.quantile(rhos,[.025,.975]).tolist() if rhos else None,
    }


def plot_results(scores, names, epsilon, samples, output, rho, provenance):
    maxima = scores.max(axis=1, keepdims=True)
    normalized = np.divide(scores, maxima, out=np.zeros_like(scores), where=maxima > 1e-12)
    ranks = np.array([station_ranks(v) for v in scores])
    order = np.argsort(-scores[0], kind='stable')
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix', 'svg.fonttype':'path', 'font.size':9,
        'axes.labelsize':10, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'axes.linewidth':.7, 'axes.labelpad':5, 'hatch.linewidth':.55,
        'xtick.direction':'out', 'ytick.direction':'out',
        'xtick.major.size':3, 'ytick.major.size':3,
        'xtick.major.width':.65, 'ytick.major.width':.65,
        'text.color':'#202020', 'axes.labelcolor':'#202020',
    })
    # Double-column width; outlined glyphs preserve the font when embedding SVG.
    fig, axes = plt.subplots(1,2,figsize=(7.16,3.0), gridspec_kw={'width_ratios':[1.28,1]})
    fig.subplots_adjust(left=.083,right=.985,bottom=.25,top=.91,wspace=.32)
    display_names = [name.replace('pv_','PV ') for name in names]
    x = np.arange(len(names))
    width = .34
    axes[0].bar(x-width/2, normalized[0,order], width, color='#376d8b', edgecolor='#244b60', linewidth=.5, label='Mechanistic OPF', zorder=2)
    axes[0].bar(x+width/2, normalized[1,order], width, color='#dfa34b', edgecolor='#624820', linewidth=.5, hatch='///', label='Jointly trained surrogate', zorder=2)
    if 'bootstrap' in provenance:
        bounds = np.asarray(provenance['bootstrap']['normalized_ci95'])
        for method, offset in enumerate((-width/2,width/2)):
            low, high = bounds[:,method,order]
            axes[0].vlines(x+offset,low,high,color='#252525',lw=.7,zorder=4)
            axes[0].hlines(low,x+offset-.04,x+offset+.04,color='#252525',lw=.7,zorder=4)
            axes[0].hlines(high,x+offset-.04,x+offset+.04,color='#252525',lw=.7,zorder=4)
    axes[0].set(xticks=x, xticklabels=np.array(display_names)[order], ylim=(0,1.10),
                yticks=np.arange(0,1.01,.2), ylabel='Normalized importance', xlabel='PV station')
    axes[0].set_xlim(-.6,len(names)-.4)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles,labels,loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,
                   frameon=False,fontsize=8.5,handlelength=1.7,handletextpad=.6,
                   columnspacing=1.5,borderaxespad=0.)
    axes[1].plot([.6,len(names)+.4],[.6,len(names)+.4],color='#929292',lw=.75,dashes=(3,3),zorder=1)
    axes[1].scatter(ranks[0],ranks[1],color='#376d8b',edgecolor='white',linewidth=.6,s=32,zorder=3)
    for k,name in enumerate(display_names):
        right_edge = ranks[0,k] >= len(names)-.25
        axes[1].annotate(name,(ranks[0,k],ranks[1,k]),xytext=(-5,7) if right_edge else (5,6),
                         ha='right' if right_edge else 'left',textcoords='offset points',fontsize=8.5)
    axes[1].set(xlim=(.6,len(names)+.4),ylim=(.6,len(names)+.4),
                xticks=np.arange(1,len(names)+1),yticks=np.arange(1,len(names)+1),
                xlabel='Rank (mechanistic OPF)',ylabel='Rank (jointly trained surrogate)')
    axes[1].set_aspect('equal',adjustable='box')
    agreement = f'Spearman $\\rho = {rho:.2f}$' if rho is not None else 'No distinct ranking (ties)'
    axes[1].text(.04,.96,agreement,transform=axes[1].transAxes,va='top',fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#e4e4e4',lw=.45)
        ax.set_axisbelow(True)
    # Put subfigure captions below the x-axis labels, as expected in papers.
    for ax, caption in zip(axes, ('(a) Station importance', '(b) Ranking agreement')):
        position = ax.get_position()
        fig.text(position.x0 + position.width / 2, .065, caption,
                 ha='center', va='center', fontsize=10)
    output.mkdir(parents=True,exist_ok=True)
    result = output/'pv_station_mechanistic_consistency.svg'
    provenance.update(importance=scores.tolist(), normalized=normalized.tolist(), ranks=ranks.tolist(), spearman=rho)
    fig.savefig(result, metadata={'Description':json.dumps(provenance)})
    plt.close(fig)
    return result, normalized, ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/ieee13.yaml')
    parser.add_argument('--data',default='data/ieee13/ieee13.npz')
    parser.add_argument('--checkpoint',default='checkpoints/dual_agent.pt')
    parser.add_argument('--epsilon',type=float,default=.05)
    parser.add_argument('--samples',type=int,default=0,help='0 uses the full test split')
    parser.add_argument('--split',choices=('validation','test'),default='test')
    parser.add_argument('--perturbation',choices=('relative-capacity','equal-power'),default='relative-capacity')
    parser.add_argument('--output-dir',default='figures/pv_station_mechanistic_consistency')
    parser.add_argument('--replot-svg',type=Path,help='Redraw stored SVG metadata without model evaluation')
    args = parser.parse_args()
    if args.replot_svg is not None:
        root = ET.parse(args.replot_svg).getroot()
        description = root.find('.//{http://purl.org/dc/elements/1.1/}description')
        if description is None or not description.text:
            parser.error('The SVG contains no experiment metadata')
        saved = json.loads(description.text)
        result,_,_ = plot_results(np.asarray(saved['importance']),saved['stations'],
                                 saved['epsilon'],saved['samples'],Path(args.output_dir),
                                 saved['spearman'],saved)
        print(f'Redrawn without evaluation: {result.resolve()}')
        return
    if not 0 < args.epsilon < 1 or args.samples < 0:
        parser.error('Require 0 < epsilon < 1 and samples >= 0')
    torch.set_num_threads(1)
    config = load_config(args.config)
    torch.manual_seed(config.seed)
    dataset = ScenarioDataset(args.data)
    validate_dataset_matches_config(dataset, config)
    _,validation,test = make_train_validation_test_loaders(dataset,batch_size=1,train_fraction=config.data_split.train_fraction,validation_fraction=config.data_split.validation_fraction,test_fraction=config.data_split.test_fraction,seed=config.seed)
    loader = validation if args.split == 'validation' else test
    f,s = build_models(config,dataset.pv_history.shape[2])
    system = DualAgentSystem(f,s)
    system.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    system.eval()
    graph = make_complete_graph(dataset.pv_history.shape[2])
    e = TorchLinDistFlowEvaluator(config.network.dss_master, config.network.pv_systems, config.network.batteries, config.lindistflow,config.problem.interval_hours).double().eval()
    opf = MechanisticOPF(e,config)
    names = list(config.network.pv_systems)
    count = min(args.samples or len(loader),len(loader))
    sensitivity, baseline_costs, violations, support_fractions = [], [], [], []
    forecast_spreads, out_of_bounds = [], []
    for n,batch in enumerate(loader):
        if n >= count:
            break
        capacity = batch['pv_history'][0,0,:,4]*1000
        if torch.any(capacity <= 0):
            raise ValueError('Synthetic dataset capacity feature must be positive')
        with torch.no_grad():
            raw = f(batch['pv_history'],graph)
        # Preserve the deployed model input, including any decision-focused bias.
        base = raw
        forecast_spreads.append(float((raw.amax(-1)-raw.amin(-1)).max()))
        out_of_bounds.append(float(((raw < 0) | (raw > capacity)).float().mean()))
        # Match future hours to the known historical clear-sky feature.
        history_hours = batch['pv_history'][0,:,0,3]*24
        future_hours = (history_hours[-1]+config.problem.interval_hours*torch.arange(1,base.shape[2]+1)) % 24
        distance = (history_hours[:,None]-future_hours[None,:]).abs()
        nearest = torch.minimum(distance,24-distance).argmin(dim=0)
        daylight = (batch['pv_history'][0,nearest,:,1] > 1e-6).float()[None,None]
        scale = capacity if args.perturbation == 'relative-capacity' else torch.full_like(capacity, float(capacity.min()))
        mask = common_perturbation_mask(base,capacity,daylight,args.epsilon,scale)
        fraction = float(mask.sum()/daylight.expand_as(base).sum().clamp_min(1))
        support_fractions.append(fraction)
        def costs(pv):
            with torch.no_grad():
                learned = s(pv,batch['load_forecast'],graph)['dispatch'].double()
            physical = opf.solve(pv,batch['load_forecast'])
            values = []
            for dispatch in (physical,learned):
                with torch.no_grad():
                    result = e(dispatch,batch['pv_target'].double().unsqueeze(1),batch['load_forecast'].double())
                values.append(float(result['operating_cost'].item()))
                violations.append([float(result[k].item()) for k in ['voltage_violation','line_flow_violation','soc_violation','terminal_soc_deviation']])
            return np.array(values)
        baseline = costs(base)
        baseline_costs.append(baseline)
        current = np.zeros((2,len(names)))
        for k in range(len(names)):
            for direction in (-1,1):
                changed = perturb(base,scale,mask,k,args.epsilon,direction)
                # Shared support and denominator across stations; no post-hoc clipping.
                if fraction > 0:
                    current[:,k] += np.abs(costs(changed)-baseline)/(2*args.epsilon*fraction)
        sensitivity.append(current)
        print(f'sample={n+1}/{count} baseline_opf={baseline[0]:.6f} baseline_surrogate={baseline[1]:.6f}',flush=True)
    scores = np.mean(sensitivity,axis=0)
    rho = ranking_agreement(scores)
    provenance = {
        'epsilon':args.epsilon, 'samples':count, 'seed':config.seed, 'split':args.split,
        'perturbation':args.perturbation,
        'rank_relative_tolerance':1e-4,
        'stations':names, 'test_indices':list(loader.dataset.indices[:count]),
        'capacity_kw':capacity.tolist(), 'mask':'shared daylight support with full symmetric headroom at every station',
        'baseline':'unaltered deployed forecaster output; never clipped',
        'importance_denominator':'2 * epsilon * common daylight support fraction',
        'support_fractions':support_fractions,
        'mean_out_of_capacity_fraction':float(np.mean(out_of_bounds)),
        'max_forecast_station_spread':max(forecast_spreads),
        'solver':'HiGHS MILP, relative gap 1e-7; hard battery power/SOC and charge-discharge exclusion; soft voltage/line/terminal penalties',
        'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'config_text':Path(args.config).read_text(),
        'per_sample_importance':np.array(sensitivity).tolist(),
        'mean_baseline_costs':np.mean(baseline_costs,axis=0).tolist(),
        'clipping_fraction':0.0,
        'max_violations':np.max(violations,axis=0).tolist(),
        'bootstrap':paired_bootstrap(sensitivity,seed=config.seed),
    }
    result, normalized, ranks = plot_results(scores,names,args.epsilon,count,Path(args.output_dir),rho,provenance)
    print('station,opf_importance,surrogate_importance,opf_normalized,surrogate_normalized,opf_rank,surrogate_rank')
    for k,name in enumerate(names):
        print(name,*scores[:,k],*normalized[:,k],*ranks[:,k],sep=',')
    print(f'spearman={rho}; mean_support_fraction={np.mean(support_fractions):.6f}; clipping_fraction=0')
    print('mean_baseline_costs=',np.mean(baseline_costs,axis=0))
    print('max_violations [voltage,line,soc,terminal]=',np.max(violations,axis=0))
    print('paired_bootstrap=',json.dumps(provenance['bootstrap']))
    print(f'output={result.resolve()}')


if __name__ == '__main__':
    main()
