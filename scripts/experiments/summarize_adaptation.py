"""Summarize paired optimization seeds without selecting seeds or checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phase3_cross_evaluation import plot_cross_matrix


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--alternating-results', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory/'results.json').read_text())
    rows = data['results']
    arms = data['arguments']['arms']
    seeds = data['arguments']['seeds']
    lookup = {(r['seed'], r['arm']):r for r in rows}
    matrix_sources = {s:args.directory for s in seeds}
    provenance = [str(args.directory/'results.json')]
    if args.alternating_results:
        replacement = {}
        for directory in args.alternating_results:
            other = json.loads((directory/'results.json').read_text())
            for key in ['train_indices','validation_indices','test_indices','checkpoints','config']:
                if other[key] != data[key]:
                    raise ValueError(f'Incompatible paired experiment: {key}')
            for key in ['forecast_epochs','surrogate_epochs','rounds','interval','dropout']:
                if other['arguments'][key] != data['arguments'][key]:
                    raise ValueError(f'Incompatible budget or protocol: {key}')
            provenance.append(str(directory/'results.json'))
            for r in other['results']:
                if r['arm'] != 'alternating':continue
                if r['seed'] in replacement:raise ValueError('Duplicate replacement seed')
                replacement[r['seed']] = r
                matrix_sources[r['seed']] = directory
        if set(replacement) != set(seeds):raise ValueError('Replacement seeds must match exactly')
        for seed,row in replacement.items():lookup[seed,'alternating']=row
    if len(lookup) != len(arms)*len(seeds):
        raise ValueError('Cannot summarize an incomplete paired experiment')
    metrics = ['operating_cost', 'objective', 'terminal_soc_deviation']
    out = args.output_dir or args.directory
    out.mkdir(parents=True,exist_ok=True)
    report = {'seeds':seeds, 'sources':provenance, 'limitation':data['limitation'],
              'arms':{}, 'paired_comparisons':{}}
    fig, axes = plt.subplots(1,3,figsize=(13,4), layout='constrained')
    for ax, metric in zip(axes, metrics):
        for i,arm in enumerate(arms):
            values = np.array([lookup[s,arm]['test'][metric] for s in seeds])
            stats = dict(mean=float(values.mean()),std=float(values.std(ddof=1)),values=values.tolist())
            report['arms'].setdefault(arm,{})[metric] = stats
            ax.errorbar(i,stats['mean'],yerr=stats['std'],fmt='o',capsize=5)
            ax.scatter(np.full(len(values),i)+np.linspace(-.08,.08,len(values)),values,s=15,alpha=.6)
        ax.set_xticks(range(len(arms)),[a.replace('_','\n') for a in arms])
        ax.set_title(metric.replace('_',' ').title())
        ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'paired_comparison.png',dpi=220)
    plt.close(fig)
    for control in ['forecast_only','forecast_budget']:
        if control not in arms:continue
        a=np.array([lookup[s,'alternating']['test']['operating_cost'] for s in seeds])
        b=np.array([lookup[s,control]['test']['operating_cost'] for s in seeds])
        report['paired_comparisons'][control]=dict(
            cost_reduction=(b-a).tolist(),relative_reduction_percent=((b-a)/b*100).tolist(),
            mean_reduction_percent=float(((b-a)/b*100).mean()),wins=int((a<b).sum()))
    matrices = np.stack([np.load(matrix_sources[s]/f'matrix_{s}.npy') for s in seeds])
    normalized = matrices / matrices[:,0,0][:,None,None]
    np.savez(out/'mean_cross_matrix.npz',raw_mean=matrices.mean(0),
             raw_std=matrices.std(0,ddof=1),normalized_mean=normalized.mean(0),
             normalized_std=normalized.std(0,ddof=1))
    plot_cross_matrix(normalized.mean(0),out/'mean_cross_matrix.png')
    (out/'aggregate.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
