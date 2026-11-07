"""Plot real CIFAR-10 JSON results produced by run_experiment.py."""
import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['fig8', 'rp', 'adc', 'ladder'], required=True)
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True, help='Output stem, without extension')
    args = p.parse_args()
    paths = sorted({Path(v) for pattern in args.input for v in glob.glob(pattern)})
    if not paths:
        p.error('No input JSON files matched')
    records = [json.loads(f.read_text(encoding='utf-8')) for f in paths]
    if any(r['synthetic'] for r in records):
        p.error('Synthetic smoke results cannot be plotted as experimental accuracy')
    ids = records[0]['indices']
    if any(r['indices'] != ids for r in records):
        p.error('Use identical image subsets within one plot')
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    if args.kind == 'fig8':
        if len(records) != 1 or 'bl' not in records[0]:
            p.error('fig8 requires exactly one BL JSON')
        rows = records[0]['bl']['by_rows']
        x = np.arange(len(rows))
        ax.bar(x-.18, [r['mean_pos_uA'] for r in rows], .36, label='Positive BL')
        ax.bar(x+.18, [r['mean_neg_uA'] for r in rows], .36, label='Negative BL')
        ax.set_xticks(x,[r['rows'] for r in rows])
        ax.set(xlabel='Logical input rows per array', ylabel='Mean LSB BL current (uA)')
    else:
        allowed = dict(rp={'rp'},adc={'adc'},ladder={'fp','w8','w8a8','q8','program','write-read','retention'})[args.kind]
        if any(r['arguments']['case'] not in allowed for r in records):
            p.error(f'{args.kind} accepts cases {sorted(allowed)}')
        order = {'fp':0,'w8':1,'w8a8':2,'q8':3,'program':4,'write-read':5,'retention':6}
        def coordinate(r):
            a=r['arguments']
            if args.kind == 'rp': return a['rp']
            if args.kind == 'adc': return a['bits']
            return order[a['case']] + (a['days']/10 if a['case']=='retention' else 0)
        records.sort(key=coordinate)
        if len({coordinate(r) for r in records}) != len(records):
            p.error('Duplicate conditions; provide one JSON per plotted point')
        for r in records:
            a=r['arguments']
            rows.append(dict(case=a['case'],x=coordinate(r),days=a['days'],N=a['n'],
                             nruns=a['nruns'],top1=r['top1_pct']['mean'],
                             std_top1=r['top1_pct']['sample_std'],top5=r['top5_pct']['mean'],
                             std_top5=r['top5_pct']['sample_std']))
        x = list(range(len(rows))) if args.kind=='ladder' else [r['x'] for r in rows]
        for metric,color in [('top1','tab:blue'),('top5','tab:orange')]:
            if args.kind == 'ladder' and metric=='top5': continue
            keep=[i for i in range(len(rows)) if args.kind!='rp' or x[i]>0]
            ax.errorbar([x[i] for i in keep],[rows[i][metric] for i in keep],
                        yerr=[rows[i]['std_'+metric] or 0 for i in keep],
                        marker='o',capsize=3,color=color,label=metric)
            if args.kind=='rp':
                for i in range(len(rows)):
                    if x[i]==0: ax.axhline(rows[i][metric],color=color,ls='--',alpha=.6,label=metric+' Rp=0')
        ax.set_ylabel('CIFAR-10 accuracy (%)')
        if args.kind=='rp':
            ax.set_xscale('log')
            ax.set_xlabel('Adjacent-cell BL resistance (ohm)')
        elif args.kind=='adc':
            ax.set_xlabel('ADC resolution (bits)')
            ax.set_xticks(x)
        else:
            names=[r['case']+(f" {r['days']:g}d" if r['case']=='retention' else '') for r in rows]
            ax.set_xticks(x,names,rotation=30,ha='right')
            ax.set_xlabel('Cumulative configuration; error bars = sample standard deviation')
    ax.grid(axis='y',alpha=.25)
    ax.legend()
    ax.set_title('ResNet14 + CIFAR-10 | '+args.kind)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'),dpi=200)
    plt.close(fig)
    with args.out.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved',args.out.with_suffix('.png'),args.out.with_suffix('.csv'))


if __name__=='__main__': main()
