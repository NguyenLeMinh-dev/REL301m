"""Export BC loss curves and paired checkpoint comparison without modifying datasets."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline-evaluation', type=Path)
    parser.add_argument('--baseline-label', default='Scratch MASAC\n50k smoke')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = args.output_dir or args.run / 'figures'
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.run / 'summary.json').read_text())
    with (args.run / 'bc_losses.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, ax in enumerate(axes):
        for name in ('training', 'validation'):
            ax.plot([int(r['epoch']) for r in rows], [float(r[f'{name}_mse_{i}']) for r in rows], label=name)
        ax.axvline(summary['best_epoch'], color='black', linestyle=':', label='selected epoch')
        ax.set(xlabel='BC epoch', title=f'Agent {i}', yscale='log')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Action MSE')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'bc_loss.{suffix}', dpi=180)
    plt.close(fig)
    if args.baseline_evaluation:
        baseline = json.loads((args.baseline_evaluation / 'summary.json').read_text())
        bc = summary['deterministic_evaluation']
        if baseline['episodes'] != bc['episodes'] or baseline['initialization_sequence_sha256'] != bc['initialization_sequence_sha256']:
            raise ValueError('Need paired baseline/BC episode count and initializations')
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [0, 1]
        ax.bar([p - .18 for p in x], [baseline['success_rate'], bc['success_rate']], .36, label='ever-success')
        ax.bar([p + .18 for p in x], [baseline['final_success_rate'], bc['final_success_rate']], .36, label='final-success')
        ax.set_xticks(x, [args.baseline_label, 'BC only'])
        ax.set(ylabel='Success rate', ylim=(0, 1), title=f'Paired {bc["episodes"]}-episode evaluation; one checkpoint each')
        ax.legend()
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'paired_checkpoint_success.{suffix}', dpi=180)
        plt.close(fig)
        comparison = dict(bc=bc, baseline=baseline, matched_initializations=True,
                          interpretation='One checkpoint each; BC has demonstration supervision; not a 3-seed RL comparison')
        (output / 'paired_comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(output.resolve())


if __name__ == '__main__':
    main()
