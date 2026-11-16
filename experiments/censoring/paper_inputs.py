"""Write the tables, figure data and quoted numbers of the paper from censoring/results/.

    python censoring/paper_inputs.py        -> paper/generated/*.tex and *.dat

Nothing in paper/main.tex is typed in by hand that this script can compute: tables are \\input and numbers in the text
are macros from numbers.tex. Run it again after any change to the results, then rebuild the paper.
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import DATASETS, derived
from study import MODELS, REFERENCE, RESULTS, TREATMENTS

V1_LOGS = ('bpi12', 'bpi17', 'traffic', 'bpi20', 'helpdesk')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'paper', 'generated')
OTHERS = [t for t in TREATMENTS if t != 'censor_aware']
TEX = {'complete_only': r'\textsc{complete}', 'open_as_negative': r'\textsc{negative}', 'mature_only': r'\textsc{mature}',
       'censor_aware': r'\textsc{window}', REFERENCE: r'\textsc{future}'}
LOG_TEX = {'bpi12': 'BPI12', 'bpi17': 'BPI17', 'traffic': 'Traffic', 'bpi20': 'BPI20', 'helpdesk': 'Helpdesk'}
MODEL_KEY = {'trees': 'Trees', 'trees150': 'TreesLong', 'logit': 'Logit', 'gru': 'Gru'}  # letters only: TeX command names
MODEL_TEX = {'trees': 'Boosted trees, 60 rounds', 'trees150': 'Boosted trees, 150 rounds', 'logit': 'Logistic regression',
             'gru': 'Recurrent network (GRU)'}
RATIO = ['$\\approx\\frac12$', '$\\approx 1$', '$\\approx 2$', '$\\approx 3$']


def read(log, kind='trees'):
    path = os.path.join(RESULTS, log + ('' if kind == 'trees' else '.' + kind) + '.jsonl')
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path)]
    lengths = sorted(DATASETS[log]['history_days'])
    for r in rows:
        r['rank'] = lengths.index(r['history_days'])  # 0..3: history about 1/2, 1, 2, 3 times the longest deadline
    return rows


def diffs(rows, t):
    return [r['brier_minus_censor_aware'][t] for r in rows if t in r['brier_minus_censor_aware']]


def tally(rows, t):
    ds = diffs(rows, t)
    return sum(d['ci95'][0] > 0 for d in ds), sum(d['ci95'][1] < 0 for d in ds), len(ds)


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float('nan')


def f(x, fmt='{:+.3f}'):
    return '--' if x is None or not np.isfinite(x) else fmt.format(x).replace('-', '$-$').replace('+', '$+$')


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as fh:
        fh.write(text)


def table(name, columns, header, body):
    write(name, '\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n'
          % (columns, header, '\n'.join(body)))


def main():
    logs = [log for log in V1_LOGS if read(log)]  # the paper of version 1 covers these five; later logs must not change it
    trees = {log: read(log) for log in logs}
    every = [r for log in logs for r in trees[log]]
    num = {}

    # --- the logs
    body = []
    for log in logs:
        cfg = DATASETS[log]
        with open(derived(log), 'rb') as fh:
            cases = pickle.load(fh)['cases']
        tests = [r['test_cases'] for r in trees[log]]
        body.append(f"{LOG_TEX[log]} & {len(cases):,} & {sum(len(c['acts']) for c in cases):,} & {', '.join(cfg['outcomes'].values())} & "
                    f"{', '.join(str(d) for d in cfg['deadlines'])} & {len(cfg['cutoffs'])} & {min(tests):,}--{max(tests):,} \\\\")
    table('tab_logs.tex', r'lrr>{\raggedright\arraybackslash}p{1.9cm}lcr', 'Log & Cases & Events & Outcomes & Deadlines & Cutoffs & Test cases', body)

    # --- how often each treatment is significantly worse or better than censor_aware
    body, total = [], defaultdict(lambda: np.zeros(3, int))
    for log in logs:
        cells = []
        for t in OTHERS:
            w, b, n = tally(trees[log], t)
            total[t] += (w, b, n)
            cells.append(f"{w} / {b} / {n}")
        body.append(f"{LOG_TEX[log]} & " + ' & '.join(cells) + ' \\\\')
    body.append('\\midrule\nAll logs & ' + ' & '.join('%d / %d / %d' % tuple(total[t]) for t in OTHERS) + ' \\\\')
    table('tab_counts.tex', 'lccc', 'Log & ' + ' & '.join(TEX[t] for t in OTHERS), body)
    for t, key in zip(OTHERS, ('complete', 'negative', 'mature')):
        num[key + 'Worse'], num[key + 'Better'], num[key + 'Total'] = (int(v) for v in total[t])
    num['settings'] = len(every)

    # --- by length of history relative to the longest deadline
    body = []
    for k in range(4):
        rs = [r for r in every if r['rank'] == k]
        cells = [f"{mean([r['share_of_train_histories_from_open_cases'] for r in rs]) * 100:.0f}\\%"]
        for t in OTHERS:
            ds = [d['mean'] for d in diffs(rs, t)]
            cells.append(f"{f(mean(ds))} ({f(float(np.median(ds)))})" if ds else 'no data')
        body.append(f"{RATIO[k]} & " + ' & '.join(cells) + ' \\\\')
        num['openShare' + 'ABCD'[k]] = f"{mean([r['share_of_train_histories_from_open_cases'] for r in rs]) * 100:.0f}"
    table('tab_ratio.tex', 'lcccc', 'History & Open cases & ' + ' & '.join(TEX[t] for t in OTHERS), body)

    # --- bias at the longest deadline, longest history
    body = []
    for log in logs:
        rs = [r for r in trees[log] if r['rank'] == 3]
        body.append(f"{LOG_TEX[log]} & " + ' & '.join(f(mean([r[t]['pred_minus_actual_any'][-1] for r in rs if r.get(t)])) for t in TREATMENTS) + ' \\\\')
    table('tab_bias.tex', 'lcccc', 'Log & ' + ' & '.join(TEX[t] for t in TREATMENTS), body)
    bias = {t: [mean([r[t]['pred_minus_actual_any'][-1] for r in trees[log] if r['rank'] == 3 and r.get(t)]) for log in logs] for t in TREATMENTS}
    num['negBiasLo'], num['negBiasHi'] = (f"{abs(v) * 100:.0f}" for v in (max(bias['open_as_negative']), min(bias['open_as_negative'])))
    num['unbiasedBiasMax'] = f"{np.ceil(max(abs(v) for t in ('censor_aware', 'mature_only') for v in bias[t]) * 100):.0f}"

    # --- other model classes
    body, signs = [], []
    for model in MODELS:
        rows = [r for log in logs for r in read(log, model)]
        if len(rows) != len(every):
            continue
        for log in logs:  # does complete_only over-predict and open_as_negative under-predict, longest history and deadline?
            rs = [r for r in rows if r['log'] == log and r['rank'] == 3]
            signs.append(mean([r['complete_only']['pred_minus_actual_any'][-1] for r in rs if r.get('complete_only')]) > 0
                         and mean([r['open_as_negative']['pred_minus_actual_any'][-1] for r in rs]) < 0)
        for k in range(4):
            num[f"negMean{'ABCD'[k]}{MODEL_KEY[model]}"] = f(mean([d['mean'] for d in diffs([r for r in rows if r['rank'] == k], 'open_as_negative')]))
        cells = [f"{mean([r['censor_aware']['brier_ident'] for r in rows]):.3f}"]
        for t, key in zip(OTHERS, ('complete', 'negative', 'mature')):
            w, b, n = tally(rows, t)
            cells.append(f"{w} / {b} / {n}")
            cells.append(f(float(np.median([d['mean'] for d in diffs(rows, t)]))))
            num[f'{key}Worse{MODEL_KEY[model]}'], num[f'{key}Better{MODEL_KEY[model]}'] = w, b
        body.append(f"{MODEL_TEX[model]} & " + ' & '.join(cells) + ' \\\\')
    num['biasSigns'], num['biasSignsTotal'] = sum(signs), len(signs)
    table('tab_models.tex', 'lccccccc', 'Model & Brier & ' + ' & '.join(f"{TEX[t]} & median" for t in OTHERS), body)

    # --- what labels from after the cutoff would add
    future = [r for log in logs for r in read(log, 'future')]
    if len(future) == len(every):
        by = {(r['log'], r['cutoff'], r['history_days']): r for r in future}
        body = []
        for k in range(4):
            rs = [r for r in every if r['rank'] == k]
            fut = [by[r['log'], r['cutoff'], r['history_days']][REFERENCE]['brier_ident'] for r in rs]
            win = [r['censor_aware']['brier_ident'] for r in rs]
            com = [r['complete_only']['brier_ident'] for r in rs if r.get('complete_only')]
            body.append(f"{RATIO[k]} & {mean(fut):.3f} & {mean(win):.3f} & {mean(com):.3f} & {f(mean(win) - mean(fut))} \\\\")
            num['futureGap' + 'ABCD'[k]] = f"{mean(win) - mean(fut):.3f}"
        table('tab_future.tex', 'lcccc', f"History & {TEX[REFERENCE]} & {TEX['censor_aware']} & {TEX['complete_only']} & "
              f"{TEX['censor_aware']} $-$ {TEX[REFERENCE]}", body)
        w, b, n = tally(future, REFERENCE)
        num['futureWorse'], num['futureBetter'], num['futureTotal'] = w, b, n  # future_labels against censor_aware
    else:
        write('tab_future.tex', '\\emph{(results not complete: run study.py with --treatments future\\_labels,censor\\_aware)}\n')

    # --- figure: Brier score relative to censor_aware, per log and history length
    short = []  # complete_only at the shortest history, one value per log
    for log in logs:
        lines = ['x ' + ' '.join(f"{k} {k}minus {k}plus" for k in ('complete', 'negative', 'mature'))]
        for k in range(4):
            rs = [r for r in trees[log] if r['rank'] == k]
            cells = []
            for t in OTHERS:
                ratio = [r[t]['brier_ident'] / r['censor_aware']['brier_ident'] for r in rs if r.get(t)]
                g = float(stats.gmean(ratio)) if ratio else None
                cells += [f"{g:.4f} {g - min(ratio):.4f} {max(ratio) - g:.4f}"] if ratio else ['nan nan nan']
                if t == 'complete_only' and k == 0:
                    short.append(g)
            lines.append(f"{k + 1} " + ' '.join(cells))
        write(f'fig_{log}.dat', '\n'.join(lines) + '\n')

    # --- numbers quoted in the text
    x = [r['share_of_train_histories_from_open_cases'] for r in every if 'complete_only' in r['brier_minus_censor_aware']]
    y = [r['brier_minus_censor_aware']['complete_only']['mean'] for r in every if 'complete_only' in r['brier_minus_censor_aware']]
    num['spearman'] = f"{stats.spearmanr(x, y).statistic:.2f}"
    long_ = {log: [r for r in trees[log] if r['rank'] == 3] for log in logs}
    rel = {log: mean([d['mean'] for d in diffs(long_[log], 'complete_only')]) / mean([r['censor_aware']['brier_ident'] for r in long_[log]])
           for log in logs}
    num['trafficFactor'] = f"{1 + rel['traffic']:.1f}"
    rest = [v for log, v in rel.items() if log != 'traffic']
    num['completeRelLo'], num['completeRelHi'] = f"{min(rest) * 100:.0f}", f"{max(rest) * 100:.0f}"
    num['matureGapLong'] = f"{mean([d['mean'] for d in diffs([r for r in every if r['rank'] == 3], 'mature_only')]):.3f}"
    num['matureGapTwo'] = f"{mean([d['mean'] for d in diffs([r for r in every if r['rank'] == 2], 'mature_only')]):.3f}"
    wins = [r for r in every if r['brier_minus_censor_aware']['open_as_negative']['ci95'][1] < 0]
    num['negBetterShort'] = sum(r['rank'] <= 1 for r in wins)
    # of those, the settings where censor_aware predicted more outcomes than followed (mean over identifiable deadlines)
    num['negBetterShift'] = sum(np.mean(r['censor_aware']['pred_minus_actual_any'][:len(r['identifiable_deadlines'])]) > 0 for r in wins)
    num['completeShortLo'], num['completeShortHi'] = f"{min(short):.1f}", f"{max(short):.1f}"
    neg = [mean([d['mean'] for d in diffs([r for r in every if r['rank'] == k], 'open_as_negative')]) for k in range(4)]
    num['negLossLo'], num['negLossHi'] = f"{min(neg):.2f}", f"{max(neg):.2f}"
    write('numbers.tex', ''.join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(num.items())))
    print(f"{len(logs)} logs, {len(every)} settings; wrote {len(os.listdir(OUT))} files to {os.path.normpath(OUT)}")
    print({k: v for k, v in sorted(num.items())})


if __name__ == '__main__':
    main()
