"""Tables, figure data and quoted numbers of the journal manuscript, from the stored result files.

    python censoring/journal_inputs.py     -> paper_journal/generated/

Nothing is trained here. Every number in paper_journal/main.tex that comes from an experiment is written by this
script, either as a table, as a data file of a figure, or as a macro in numbers.tex. The paper of version 1
(paper/) has its own generator, paper_inputs.py, and is not touched.
"""
import csv
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auc_summary
import calibration
import pooled
from datasets import DATASETS, derived
from decisions import rank_of
from study import RESULTS, TREATMENTS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'paper_journal')
OUT = os.path.join(ROOT, 'generated')
OTHERS = [t for t in TREATMENTS if t != 'censor_aware']
TEX = {'complete_only': r'\tcomplete', 'open_as_negative': r'\tnegative', 'mature_only': r'\tmature', 'censor_aware': r'\twindow'}
KEY = {'complete_only': 'complete', 'open_as_negative': 'negative', 'mature_only': 'mature', 'censor_aware': 'window'}
LOG_TEX = {'bpi12': 'BPI12', 'bpi17': 'BPI17', 'traffic': 'Traffic', 'bpi20': 'BPI20d', 'helpdesk': 'Helpdesk', 'bpi14': 'BPI14',
           'bpi20r': 'BPI20r', 'bpi15_1': 'BPI15-1', 'bpi15_3': 'BPI15-3', 'bpi15_4': 'BPI15-4', 'bpi15_5': 'BPI15-5'}
PROCESS = {'bpi12': 'loan applications, bank', 'bpi17': 'loan applications, same bank', 'traffic': 'traffic fines, local police',
           'bpi20': 'travel expense claims, university', 'helpdesk': 'help desk tickets, software firm',
           'bpi14': 'IT incidents, service desk of a bank', 'bpi20r': 'requests for payment, same university',
           'bpi15_1': 'building permits, municipality 1', 'bpi15_3': 'building permits, municipality 3',
           'bpi15_4': 'building permits, municipality 4', 'bpi15_5': 'building permits, municipality 5'}
HISTORY = [r'$\approx\frac12$', r'$\approx 1$', r'$\approx 2$', r'$\approx 3$']
LETTER = 'ABCD'  # history lengths in macro names: TeX command names hold letters only
LOGS = list(DATASETS)
num = {}


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as fh:
        fh.write(text)


def table(name, columns, header, body):
    write(name, '\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n' % (columns, header, '\n'.join(body)))


def pc(x, sign=True):
    """A log ratio as a percentage: no decimals from 10% on, one below."""
    v = (np.exp(x) - 1) * 100
    s = f"{v:+.0f}" if abs(v) >= 9.95 else f"{v:+.1f}"
    if float(s) == 0:
        return '0.0'
    return s if sign else s.lstrip('+')


def tex(s):
    """Numbers with a sign go into math mode, so that the minus is a minus."""
    return f"${s}$"


def effect(c, logs=True):
    return f"{tex(pc(c['mean']) + r'\%')} ({tex(pc(c['ci'][0], False))} to {tex(pc(c['ci'][1], False))})" + (f", {c['sign'][0]}" if logs else '')


def mean_ci(values):
    v = np.asarray(list(values), dtype=float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def plain(x, digits=1, sign=False):
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return f"{0:.{digits}f}" if float(s) == 0 else s


def signed(m, lo, hi, digits=1):
    return f"{tex(plain(m, digits, True))} ({tex(plain(lo, digits))} to {tex(plain(hi, digits))})"


def lines(log, suffix):
    path = os.path.join(RESULTS, f'{log}.{suffix}.jsonl')
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def logs_table():
    body = []
    for log in LOGS:
        cfg = DATASETS[log]
        with open(derived(log), 'rb') as fh:
            cases = pickle.load(fh)['cases']
        tests = [r['test_cases'] for r in pooled.read(log)]
        body.append(f"{LOG_TEX[log]} & {PROCESS[log]} & {len(cases):,} & {', '.join(dict.fromkeys(cfg['outcomes'].values()))} & "
                    f"{', '.join(f'{d:g}' for d in cfg['deadlines'])} & {', '.join(f'{d:g}' for d in sorted(cfg['history_days']))} & "
                    f"{len(cfg['cutoffs'])} & {min(tests):,}--{max(tests):,} \\\\")
        num['cases' + LOG_TEX[log].replace('-', '').replace('1', 'One').replace('2', 'Two').replace('3', 'Three').replace('4', 'Four').replace('5', 'Five').replace('7', 'Seven').replace('0', 'Zero')] = f"{len(cases):,}"
    table('tab_logs.tex', r'l>{\raggedright\arraybackslash}p{3.1cm}r>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{2.1cm}>{\raggedright\arraybackslash}p{2.3cm}cr',
          'Log & Process & Cases & Outcomes & Deadlines & Histories & Cutoffs & Test cases', body)
    num['logs'], num['organisations'] = len(LOGS), len({DATASETS[log].get('org', log) for log in LOGS})


def treatments():
    names = {'trees': 'Boosted trees', 'logit': 'Logistic regression', 'gru': 'Recurrent network'}
    body, every = [], {}
    for model, title in names.items():
        cells = pooled.pool(LOGS, OTHERS, model)
        every[model] = cells
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for t in OTHERS:
            body.append(f"{TEX[t]} & " + ' & '.join(effect(cells[t, k]) if (t, k) in cells else 'no data' for k in range(4)) + ' \\\\')
        if model != 'gru':
            body.append('\\addlinespace')
    table('tab_pooled.tex', 'lcccc', 'Treatment & ' + ' & '.join(HISTORY), body)
    trees = every['trees']
    for t in OTHERS:
        for k in range(4):
            if (t, k) in trees:
                c = trees[t, k]
                num[f'{KEY[t]}Pooled{LETTER[k]}'] = pc(c['mean'], False)
                num[f'{KEY[t]}Plain{LETTER[k]}'] = pc(c['plain'][0], False)
                num[f'{KEY[t]}Logs{LETTER[k]}'] = c['sign'][0]
                num[f'{KEY[t]}Holm{LETTER[k]}'] = f"{c['p_holm']:.3f}"
    num['signP'] = f"{trees['complete_only', 0]['sign'][2]:.3f}"
    num['completeHolmA'] = f"{trees['complete_only', 0]['p_holm']:.3f}"
    for model, key in (('logit', 'Logit'), ('gru', 'Gru')):
        num[f'completePooledA{key}'], num[f'completePooledD{key}'] = pc(every[model]['complete_only', 0]['mean'], False), pc(every[model]['complete_only', 3]['mean'], False)
    v1 = pooled.pool([log for log in LOGS if DATASETS[log]['phase'] == '1'], OTHERS, 'trees')
    num['signPFive'] = f"{v1['complete_only', 0]['sign'][2]:.2f}"
    by_org = pooled.pool(LOGS, OTHERS, 'trees', by='org')
    num['orgUnits'] = by_org['complete_only', 0]['units']
    num['orgAgree'] = sum(c['sign'][0] == c['units'] for c in by_org.values())
    num['orgCells'] = len(by_org)
    num['orgNegativeA'] = by_org['open_as_negative', 0]['sign'][0]

    # settings in which a treatment is significantly worse or better than the window rule
    body = []
    for model, title in names.items():
        rows, cells = [r for log in LOGS for r in pooled.read(log, model)], []
        for t in OTHERS:
            ds = [r['brier_minus_censor_aware'][t] for r in rows if t in r['brier_minus_censor_aware']]
            w, b = sum(d['ci95'][0] > 0 for d in ds), sum(d['ci95'][1] < 0 for d in ds)
            cells.append(f"{w} / {b} / {len(ds)}")
            if model == 'trees':
                num[KEY[t] + 'Worse'], num[KEY[t] + 'Better'], num[KEY[t] + 'Total'] = w, b, len(ds)
        body.append(f"{title} & " + ' & '.join(cells) + ' \\\\')
        if model == 'trees':
            num['settings'] = len(rows)
    table('tab_counts.tex', 'lccc', 'Model & ' + ' & '.join(TEX[t] for t in OTHERS), body)

    # effect per log, for the figure: ratio of Brier scores with 1.96 standard errors of the log's mean
    for k in range(4):
        out = ['i log complete cminus cplus negative nminus nplus']
        for i, log in enumerate(LOGS, 1):
            cells = [str(i), LOG_TEX[log]]
            for t in ('complete_only', 'open_as_negative'):
                e = pooled.log_effect(pooled.read(log), t, k)
                m, lo, hi = (np.exp(e[0]), np.exp(e[0] - 1.96 * e[1]), np.exp(e[0] + 1.96 * e[1])) if e else (np.nan,) * 3
                cells += [f"{m:.4f}", f"{m - lo:.4f}", f"{hi - m:.4f}"]
            out.append(' '.join(cells))
        write(f'fig_effects_{LETTER[k]}.dat', '\n'.join(out) + '\n')
    worst = pooled.pool(LOGS, ['complete_only'], 'trees')['complete_only', 3]['per_unit']
    num['trafficCompleteD'] = pc(worst['traffic'], False)
    num['completeWithinFive'] = sum(abs(np.exp(v) - 1) < 0.05 for v in worst.values())
    rest = sorted(100 * (np.exp(v) - 1) for u, v in worst.items() if u != 'traffic' and abs(np.exp(v) - 1) >= 0.05)
    num['completeRestCount'], num['completeRestLo'], num['completeRestHi'] = len(rest), f"{rest[0]:.0f}", f"{rest[-1]:.0f}"
    # labels from after the cutoff, run on the five logs of the first version only
    v1 = [log for log in LOGS if DATASETS[log]['phase'] == '1']
    future = {(r['log'], r['cutoff'], r['history_days']): r for log in v1 for r in lines(log, 'future')}
    for k in range(4):
        gaps = [r['censor_aware']['brier_ident'] - future[r['log'], r['cutoff'], r['history_days']]['future_labels']['brier_ident']
                for log in v1 for r in pooled.read(log) if r['rank'] == k and (r['log'], r['cutoff'], r['history_days']) in future]
        num[f'futureGap{LETTER[k]}'] = f"{np.mean(gaps):.3f}"


def pool_units(units, ratio):
    """Pooled like pooled.pool, for any grouping of logs into units and any log ratio per setting.
    units: {unit: [logs]}; ratio(row) -> log of a ratio of scores, or None. -> {rank: cell}"""
    cells = {}
    for k in range(4):
        y, se = [], []
        for members in units.values():
            es = []
            for log in members:
                v = [x for x in (ratio(r) for r in pooled.read(log) if r['rank'] == k) if x is not None]
                if len(v) >= 2:
                    es.append((np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))))
            if es:
                y.append(np.mean([e[0] for e in es]))
                se.append(np.sqrt(np.sum([e[1] ** 2 for e in es])) / len(es))
        if len(y) >= 2:
            cells[k] = {**pooled.random_effects(y, se), 'sign': pooled.sign_test(y), 'y': y}
    return cells


def robustness():
    """What the pooled comparison of complete cases only rests on: its intervals, and four other ways of taking it.
    The last four were added on 2026-11-15, after the results, when the manuscript was checked (PROTOCOL_B.md, 4h)."""
    def versus(treat, base='censor_aware', key='brier_ident'):
        return lambda r: np.log(r[treat][key] / r[base][key]) if r.get(treat) and r.get(base) else None

    def put(name, cells, ranks=range(4)):
        for k in ranks:
            c = cells[k]
            num[f'{name}{LETTER[k]}'] = pc(c['mean'], False)
            num[f'{name}Lo{LETTER[k]}'], num[f'{name}Hi{LETTER[k]}'] = pc(c['ci'][0], False), pc(c['ci'][1], False)
            num[f'{name}Logs{LETTER[k]}'] = c['sign'][0]

    main = pooled.pool(LOGS, ['complete_only'], 'trees')
    for k in range(4):                                     # required by the protocol, section 3
        c = main['complete_only', k]
        num[f'completeLo{LETTER[k]}'], num[f'completeHi{LETTER[k]}'] = pc(c['ci'][0], False), pc(c['ci'][1], False)
        num[f'completePredLo{LETTER[k]}'], num[f'completePredHi{LETTER[k]}'] = pc(c['prediction'][0], False), pc(c['prediction'][1], False)
        num[f'completeTau{LETTER[k]}'] = f"{np.sqrt(c['tau2']):.2f}"
    single = {log: [log] for log in LOGS}
    first = {log: [log] for log in LOGS if DATASETS[log]['phase'] == '1'}
    rule = {log: [log] for log in LOGS if DATASETS[log]['phase'] != '1'}
    num['firstLogs'], num['ruleLogs'] = len(first), len(rule)
    put('firstComplete', pool_units(first, versus('complete_only')))
    put('ruleComplete', pool_units(rule, versus('complete_only')))
    merged = {}
    for log in LOGS:                                       # the four permit logs are one process
        merged.setdefault('bpi15' if log.startswith('bpi15') else log, []).append(log)
    cells = pool_units(merged, versus('complete_only'))
    put('mergedComplete', cells)
    num['mergedUnits'], num['mergedSignP'] = cells[0]['units'], f"{cells[0]['sign'][2]:.3f}"
    put('allComplete', pool_units(single, versus('complete_only', key='brier_all_deadlines')), ranks=(0, 1))
    put('allNegative', pool_units(single, versus('open_as_negative', key='brier_all_deadlines')), ranks=(0, 1))
    cells = pool_units(single, versus('complete_only', base='mature_only'))
    put('overMature', cells, ranks=(2, 3))
    for k in (2, 3):
        num[f'overMatureTraffic{LETTER[k]}'] = pc(cells[k]['y'][LOGS.index('traffic')], False)
    rest = [v for k in (2, 3) for i, v in enumerate(cells[k]['y']) if LOGS[i] != 'traffic']
    num['overMatureRestLo'], num['overMatureRestHi'] = pc(min(rest), False), pc(max(rest), False)
    by_org = {}
    for log in LOGS:                                       # the sensitivity analysis of the protocol, as estimates
        by_org.setdefault(DATASETS[log].get('org', log), []).append(log)
    put('orgComplete', pool_units(by_org, versus('complete_only')))


def auc():
    data = {log: pooled.read(log) for log in LOGS}
    body = []
    for k in range(4):
        cells = []
        for key in ('auc_any_mid', 'brier_ident'):
            for t in OTHERS:
                pairs = [p for p in (auc_summary.per_log(rows, t, k, key) for rows in data.values()) if p]
                if not pairs:
                    cells.append('no data')
                    continue
                m, lo, hi = mean_ci(100 * (a - b) for a, b in pairs)
                cells.append(signed(m, lo, hi))
                num[f"{'auc' if key == 'auc_any_mid' else 'brier'}{KEY[t].capitalize()}{LETTER[k]}"] = f"{abs(m):.1f}"
        body.append(f"{HISTORY[k]} & " + ' & '.join(cells) + ' \\\\')
    table('tab_auc.tex', 'lcccccc', r' & \multicolumn{3}{c}{AUC, difference in points} & \multicolumn{3}{c}{Brier score, difference in points} \\' + '\n'
          r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}' + '\nHistory & ' + ' & '.join(TEX[t] for t in OTHERS) + ' & ' + ' & '.join(TEX[t] for t in OTHERS), body)


def methods():
    trees = pooled.pool(LOGS, ['ipcw', 'ipcw_x', 'pseudo', 'cox_trees'], 'trees', with_methods=True)
    linear = pooled.pool(LOGS, ['cox', 'dfm', 'pu'], 'linear')
    title = {'ipcw': 'Censoring weights', 'ipcw_x': 'Censoring weights by time in case$^{*}$', 'pseudo': 'Pseudo-labels', 'cox_trees': 'Cox regression, boosted trees',
             'cox': 'Cox regression', 'dfm': 'Delayed-feedback model', 'pu': 'Positive-unlabelled learning'}
    body = [r'\multicolumn{5}{l}{\emph{Against the window rule with boosted trees}} \\']
    for m in ('ipcw', 'ipcw_x', 'pseudo', 'cox_trees'):
        if (m, 0) in trees:
            body.append(f"{title[m]} & " + ' & '.join(effect(trees[m, k]) for k in range(4)) + ' \\\\')
    body += ['\\addlinespace', r'\multicolumn{5}{l}{\emph{Against the window rule with logistic regression}} \\']
    for m in ('cox', 'dfm', 'pu'):
        body.append(f"{title[m]} & " + ' & '.join(effect(linear[m, k]) for k in range(4)) + ' \\\\')
    table('tab_methods.tex', 'lcccc', 'Method & ' + ' & '.join(HISTORY), body)
    for cells, names in ((trees, ('ipcw', 'ipcw_x', 'pseudo', 'cox_trees')), (linear, ('cox', 'dfm', 'pu'))):
        for m in names:
            for k in range(4):
                if (m, k) in cells:
                    key = m.replace('_x', 'Time').replace('_trees', 'Trees')
                    num[f'{key}Pooled{LETTER[k]}'] = pc(cells[m, k]['mean'])
                    num[f'{key}Plain{LETTER[k]}'] = pc(cells[m, k]['plain'][0])
                    num[f'{key}Higher{LETTER[k]}'], num[f'{key}Lower{LETTER[k]}'] = cells[m, k]['sign'][0], cells[m, k]['sign'][1]
                    num[f'{key}Holm{LETTER[k]}'] = f"{cells[m, k]['p_holm']:.2f}"
    for m, key in (('ipcw', 'ipcw'), ('cox_trees', 'coxTrees')):
        if (m, 0) in trees and all('traffic' in trees[m, k]['per_unit'] for k in range(4)):
            traffic = [trees[m, k]['per_unit']['traffic'] for k in range(4)]
            num[key + 'TrafficLo'], num[key + 'TrafficHi'] = pc(min(traffic), False), pc(max(traffic), False)
    num['coxTrafficMax'] = pc(max(abs(linear['cox', k]['per_unit']['traffic']) for k in range(4)), False)
    return trees, linear


def tuning():
    names = {'complete_only': r'\tcomplete, tuned', 'open_as_negative': r'\tnegative, tuned', 'mature_only': r'\tmature, tuned',
             'censor_aware_fixed': r'\twindow{} with the fixed settings', 'censor_aware_rounds': r'\twindow{} with rounds tuned only'}
    body = []
    for model, title, extra in (('tuned', 'Number of rounds tuned', ['censor_aware_fixed']), ('tuned_wide', 'Rounds, learning rate and leaves tuned', ['censor_aware_rounds', 'censor_aware_fixed'])):
        logs = [log for log in LOGS if lines(log, model)]
        if len(logs) < len(LOGS):
            continue
        cells = pooled.pool(logs, OTHERS + extra, model)
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for t in OTHERS + extra:
            body.append(f"{names[t]} & " + ' & '.join(effect(cells[t, k], logs=t in OTHERS) if (t, k) in cells else 'no data' for k in range(4)) + ' \\\\')
        body.append('\\addlinespace')
        key = 'tuned' if model == 'tuned' else 'wide'
        for t in OTHERS + extra:
            for k in range(4):
                if (t, k) in cells:
                    name = KEY.get(t, t.replace('censor_aware_', ''))
                    num[f'{key}{name.capitalize()}{LETTER[k]}'] = pc(cells[t, k]['mean'], False)
                    num[f'{key}{name.capitalize()}Logs{LETTER[k]}'] = cells[t, k]['sign'][0]
        rows = [r for log in logs for r in lines(log, model)]
        chosen = [r['censor_aware']['rounds'] for r in rows if r.get('censor_aware')]
        num[key + 'BelowSixty'] = f"{100 * np.mean([c < 60 for c in chosen]):.0f}"
        num[key + 'Settings'] = len(rows)
        if model == 'tuned_wide':
            small = [r['censor_aware']['leaves'] == 7 for r in rows if r.get('censor_aware')]
            num['wideSmallTrees'] = f"{100 * np.mean(small):.0f}"
    if body:
        table('tab_tuned.tex', 'lcccc', 'Compared with the tuned \\twindow{} & ' + ' & '.join(HISTORY), body[:-1])
    # the hold-out from the recent half of the history against the random one: plain mean over logs
    plain = pooled.pool(LOGS, ['censor_aware_prospective'], 'tuned')
    for k in range(4):
        m, (lo, hi) = plain['censor_aware_prospective', k]['plain']
        num[f'prospective{LETTER[k]}'] = f"${pc(m)}\\%$ (${pc(lo, False)}$ to ${pc(hi, False)}$)"
    rows = [r for log in LOGS for r in lines(log, 'tuned')]
    num['prospectiveSame'] = sum(r['censor_aware']['rounds'] == r['censor_aware_prospective']['rounds'] for r in rows if r.get('censor_aware_prospective'))
    num['prospectiveBelowSixty'] = f"{100 * np.mean([r['censor_aware_prospective']['rounds'] < 60 for r in rows if r.get('censor_aware_prospective')]):.0f}"


def decisions():
    rows = {log: lines(log, 'decisions') for log in LOGS}

    def per_log(t, k, f=lambda x: x):
        out = {}
        for log, rs in rows.items():
            v = [f(r[t]['planning_error_per_100']) for r in rs if rank_of(r) == k and r.get(t)]
            if v:
                out[log] = float(np.mean(v))
        return out

    body = []
    for k in range(4):
        cells = []
        for t in TREATMENTS:
            v = per_log(t, k)
            cells.append(signed(*mean_ci(v.values())) if v else 'no data')
            if v:
                num[f'plan{KEY[t].capitalize()}{LETTER[k]}'] = f"{np.mean(list(v.values())):+.1f}"
                num[f'planSize{KEY[t].capitalize()}{LETTER[k]}'] = f"{abs(np.mean(list(v.values()))):.1f}"
        body.append(f"{HISTORY[k]} & " + ' & '.join(cells) + ' \\\\')
        base = per_log('censor_aware', k, abs)
        for t in OTHERS:
            v = per_log(t, k, abs)
            if v:
                num[f'planFurther{KEY[t].capitalize()}{LETTER[k]}'] = sum(v[log] > base[log] for log in v)
        num[f'planAbsWindow{LETTER[k]}'] = f"{np.mean(list(base.values())):.1f}"
    table('tab_planning.tex', 'lcccc', 'History & ' + ' & '.join(TEX[t] for t in TREATMENTS), body)

    # value forecasts at the longest history, on the logs with an amount
    value = {}
    for log, rs in rows.items():
        v = {t: [r[t]['value_ratio'] for r in rs if rank_of(r) == 3 and r.get(t) and r[t].get('value_ratio') is not None] for t in TREATMENTS}
        if v['censor_aware']:
            value[log] = {t: (float(np.mean(x)), float(np.mean(np.abs(x)))) for t, x in v.items()}
    num['valueLogs'] = len(value)
    num['valueCloserComplete'] = sum(v['censor_aware'][1] < v['complete_only'][1] for v in value.values())
    num['valueCloserNegative'] = sum(v['censor_aware'][1] < v['open_as_negative'][1] for v in value.values())
    others = [v for log, v in value.items() if log != 'traffic']
    for t in ('complete_only', 'open_as_negative', 'censor_aware'):
        signed_ = sorted(100 * v[t][0] for v in others)
        num[f'value{KEY[t].capitalize()}Lo'], num[f'value{KEY[t].capitalize()}Hi'] = plain(signed_[0], 0, True), plain(signed_[-1], 0, True)

    replay = {log: lines(log, 'replay') for log in LOGS}
    months = sum(len(rs) for rs in replay.values())
    num['replayMonths'] = months
    absolute = {t: {log: float(np.mean([abs(r[t]['planning_error_per_100']) for r in rs if r.get(t)])) for log, rs in replay.items()} for t in TREATMENTS}
    signed_ = {t: float(np.mean([np.mean([r[t]['planning_error_per_100'] for r in rs if r.get(t)]) for rs in replay.values()])) for t in TREATMENTS}
    every = [r for rs in replay.values() for r in rs]
    body = [r'Mean absolute error per 100 cases & ' + ' & '.join(f"{np.mean(list(absolute[t].values())):.1f}" for t in TREATMENTS) + r' \\',
            r'Mean signed error per 100 cases & ' + ' & '.join(tex(plain(signed_[t], 1, True)) for t in TREATMENTS) + r' \\',
            r'Months in which \twindow{} is closer & ' + ' & '.join(
                str(sum(abs(r['censor_aware']['planning_error_per_100']) < abs(r[t]['planning_error_per_100']) for r in every if r.get(t))) for t in OTHERS) + r' & \\',
            r'Logs on which \twindow{} is closer on average & ' + ' & '.join(str(sum(absolute['censor_aware'][log] < absolute[t][log] for log in LOGS)) for t in OTHERS) + r' & \\',
            r'Mean over logs: error above that of \twindow{} & ' + ' & '.join(
                signed(*mean_ci(absolute[t][log] - absolute['censor_aware'][log] for log in LOGS)) for t in OTHERS) + r' & \\']
    table('tab_replay.tex', r'>{\raggedright\arraybackslash}p{5.2cm}cccc', ' & ' + ' & '.join(TEX[t] for t in TREATMENTS), body)
    for t in TREATMENTS:
        num['replayAbs' + KEY[t].capitalize()] = f"{np.mean(list(absolute[t].values())):.1f}"
    for log, key in (('bpi17', 'BpiSeventeen'), ('traffic', 'Traffic'), ('bpi14', 'BpiFourteen')):
        rs = replay[log]
        per_week = lambda r: r['test_cases'] / (30 / 7)
        num['week' + key] = f"{np.mean([per_week(r) for r in rs]):,.0f}"
        for t in ('complete_only', 'open_as_negative', 'censor_aware'):
            num[f'week{key}{KEY[t].capitalize()}'] = f"{np.mean([abs(r[t]['planning_error_per_100']) / 100 * per_week(r) for r in rs if r.get(t)]):.0f}"
    # one log month by month, for the figure
    out = ['month ' + ' '.join(KEY[t] for t in TREATMENTS)]
    for i, r in enumerate(replay['traffic'], 1):
        out.append(f"{i} " + ' '.join(f"{r[t]['planning_error_per_100']:.2f}" if r.get(t) else 'nan' for t in TREATMENTS))
    write('fig_replay_traffic.dat', '\n'.join(out) + '\n')
    num['replayTrafficMonths'] = len(replay['traffic'])
    odd = [r for r in replay['traffic'] if r['censor_aware']['planning_error_per_100'] < -25]   # months that no treatment anticipates
    num['replayOddMonths'] = len(odd)
    num['replayOddLo'], num['replayOddHi'] = f"{min(r['actual_per_100'] for r in odd):.0f}", f"{max(r['actual_per_100'] for r in odd):.0f}"
    num['replayOddMedian'] = f"{np.median([r['actual_per_100'] for r in replay['traffic']]):.0f}"
    num['replayOddCompleteCloser'] = sum(abs(r['complete_only']['planning_error_per_100']) < abs(r['censor_aware']['planning_error_per_100']) for r in odd)
    num['replayTrafficFirst'], num['replayTrafficLast'] = replay['traffic'][0]['cutoff'][:7], replay['traffic'][-1]['cutoff'][:7]


def calibration_detail():
    rows = calibration.read()
    if len(rows) < len(LOGS):
        return
    out = ['bin ' + ' '.join(f"{KEY[t]}p {KEY[t]}y {KEY[t]}w" for t in TREATMENTS)]
    cols = {}
    for t in TREATMENTS:
        parts = [calibration.pooled_bins(rs, t) for rs in rows.values()]
        weight = np.mean([c / c.sum() for c, _, _ in parts], axis=0)
        ps, ys = [], []
        for b in range(calibration.BINS):
            use = [(c[b] / c.sum(), f[b], h[b]) for c, f, h in parts if c[b]]
            s = sum(u[0] for u in use)
            ps.append(sum(u[0] * u[1] for u in use) / s if use else np.nan)
            ys.append(sum(u[0] * u[2] for u in use) / s if use else np.nan)
        cols[t] = (ps, ys, weight)
        num['gap' + KEY[t].capitalize()] = f"{100 * np.nansum(weight * np.abs(np.array(ps) - np.array(ys))):.1f}"
        num['top' + KEY[t].capitalize() + 'Share'], num['top' + KEY[t].capitalize() + 'Happened'] = f"{100 * weight[-1]:.0f}", f"{100 * ys[-1]:.0f}"
        use = [np.mean([r[t][max(r[t], key=float)]['any']['reliability'] for r in rs if rank_of(r) == 3 and r.get(t)]) for rs in rows.values()]
        num['reliability' + KEY[t].capitalize()] = f"{np.mean(use):.3f}"
    num['negativeQuarterHappened'] = f"{100 * cols['open_as_negative'][1][2]:.0f}"
    for b in range(calibration.BINS):
        out.append(f"{b + 1} " + ' '.join(f"{cols[t][0][b]:.4f} {cols[t][1][b]:.4f} {cols[t][2][b]:.4f}" for t in TREATMENTS))
    write('fig_reliability.dat', '\n'.join(out) + '\n')

    # slope by deadline (shortest and longest of those shorter than the history), any outcome; mean over logs
    body = []
    for k in range(4):
        cells = []
        for t in TREATMENTS:
            short, long_ = [], []
            for rs in rows.values():
                s = [r[t][min(r[t], key=float)]['any']['slope'] for r in rs if rank_of(r) == k and r.get(t)]
                g = [r[t][max(r[t], key=float)]['any']['slope'] for r in rs if rank_of(r) == k and r.get(t)]
                s, g = [v for v in s if v is not None], [v for v in g if v is not None]
                if s and g:
                    short.append(np.mean(s)); long_.append(np.mean(g))
            cells.append(f"{np.mean(short):.2f}, {np.mean(long_):.2f}" if short else 'no data')
            if short:
                num[f'slope{KEY[t].capitalize()}{LETTER[k]}'] = f"{np.mean(long_):.2f}"
                num[f'slopeShort{KEY[t].capitalize()}{LETTER[k]}'] = f"{np.mean(short):.2f}"
        body.append(f"{HISTORY[k]} & " + ' & '.join(cells) + ' \\\\')
    table('tab_slope.tex', 'lcccc', 'History & ' + ' & '.join(TEX[t] for t in TREATMENTS), body)

    # the money case on the traffic fines
    body = []
    for deadline in ('90', '365'):
        for k in range(4):
            rs = [r for r in rows['traffic'] if rank_of(r) == k and r.get('censor_aware') and deadline in r['censor_aware']]
            if not rs:
                continue
            cells = [f"{deadline} days", HISTORY[k], f"{np.mean([r['censor_aware'][deadline]['realised_value'] for r in rs]) / 1000:,.0f}"]
            for t in TREATMENTS:
                use = [r[t][deadline] for r in rs if r.get(t) and deadline in r[t]]
                if use:
                    m, lo, hi = mean_ci(100 * (u['forecast_value'] / u['realised_value'] - 1) for u in use)
                    cells.append(f"{tex(plain(m, 0, True))}\\% ({tex(plain(lo, 0))} to {tex(plain(hi, 0))})")
                    num[f"money{KEY[t].capitalize()}{'Short' if deadline == '90' else 'Long'}{LETTER[k]}"] = plain(m, 0, True)
                else:
                    cells.append('no data')
            body.append(' & '.join(cells) + ' \\\\')
        if deadline == '90':
            body.append('\\addlinespace')
    table('tab_money.tex', 'llrcccc', 'Deadline & History & Realised & ' + ' & '.join(TEX[t] for t in TREATMENTS), body)


def anchor():
    path = os.path.join(RESULTS, 'anchor.json')
    if not os.path.exists(path):
        return
    res = json.load(open(path))
    body = []
    for name, title in (('traffic', 'Traffic fines'), ('bpic2012_1', 'BPI 2012, accepted')):
        r = res[name]
        key = 'Traffic' if name == 'traffic' else 'Loans'
        num[f'anchor{key}Published'] = f"{r['published_auc']:.2f}"
        for kind, label in (('benchmark', 'labels of completed traces'), ('strict', 'completed cases only')):
            m = r[kind]['study settings']
            v = lambda k, d=3: f"{m[k]['value']:.{d}f} ({m[k]['ci95'][0]:.{d}f} to {m[k]['ci95'][1]:.{d}f})"
            body.append(f"{title if kind == 'benchmark' else ''} & {label} & {r[kind]['train_cases']:,} & {r[kind]['positive_share_train_cases']:.2f} & {v('auc_weighted_by_prefix_length')} & {v('brier')} & "
                        f"{tex(format(m['forecast_minus_positive_share']['value'], '+.3f'))} \\\\")
            tag = 'Benchmark' if kind == 'benchmark' else 'Strict'
            num[f'anchor{key}{tag}Auc'] = f"{m['auc_weighted_by_prefix_length']['value']:.3f}"
            num[f'anchor{key}{tag}Brier'] = f"{m['brier']['value']:.3f}"
            num[f'anchor{key}{tag}Bias'] = f"{100 * m['forecast_minus_positive_share']['value']:+.1f}"
            num[f'anchor{key}{tag}Cases'] = f"{r[kind]['train_cases']:,}"
            num[f'anchor{key}{tag}Share'] = f"{100 * r[kind]['positive_share_train_cases']:.0f}"
            num[f'anchor{key}{tag}AucLong'] = f"{r[kind]['500 rounds']['auc_weighted_by_prefix_length']['value']:.3f}"
        num[f'anchor{key}TestShare'] = f"{100 * r['positive_share_test_cases']:.0f}"
        still_open = r['benchmark']['train_cases'] - r['strict']['train_cases']
        num[f'anchor{key}Open'], num[f'anchor{key}OpenShare'] = f"{still_open:,}", f"{100 * still_open / r['benchmark']['train_cases']:.0f}"
        num[f'anchor{key}Gap'] = f"{abs(r['benchmark']['study settings']['auc_weighted_by_prefix_length']['value'] - r['published_auc']):.3f}"
        if name == 'traffic':
            body.append('\\addlinespace')
    table('tab_anchor.tex', 'llrcccc', 'Dataset & Training cases keep & Cases & Positive & AUC & Brier score & Bias', body)


def prevalence():
    path = os.path.join(ROOT, 'audit', 'protocols.csv')
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    count = lambda c: sum(r['category'] == c for r in rows)
    read = [r for r in rows if r['category'] not in ('unread', 'out_of_scope')]
    names = [('random', 'Random division of cases, or cross-validation'), ('temporal_completed', 'Split in time; training cases keep the label of the completed trace'),
             ('strict_temporal', 'Split in time; only cases completed before the split train'), ('open_aware', 'Open cases train, and their incompleteness is modelled'),
             ('not_stated', 'Not stated')]
    body = [f"{title} & {count(c)} & {100 * count(c) / len(read):.0f}\\% \\\\" for c, title in names]
    body.append(f"\\midrule\nRead and in scope & {len(read)} & \\\\")
    table('tab_prevalence.tex', r'>{\raggedright\arraybackslash}p{9cm}rr', 'How the training set is built & Papers & Share', body)
    num['auditPapers'], num['auditRead'], num['auditUnread'], num['auditOutOfScope'] = len(rows), len(read), count('unread'), count('out_of_scope')
    for c, key in (('random', 'Random'), ('temporal_completed', 'Temporal'), ('strict_temporal', 'Strict'), ('open_aware', 'Aware'), ('not_stated', 'Unstated')):
        num['audit' + key] = count(c)
    num['auditRemoved'] = sum(r['incomplete_removed'] == 'yes' for r in read)
    num['auditByReference'] = sum(r['follows_benchmark'] == 'yes' for r in read)
    # the random draw of the second check; papers read later from files saved by hand are marked 'hand: ...'
    drawn = [r for r in rows if r.get('second_check') and not r['second_check'].startswith('hand')]
    checked = [r for r in drawn if r['second_check'] != 'not retrieved']
    num['auditDrawn'], num['auditChecked'], num['auditAgreed'] = len(drawn), len(checked), sum(r['second_check'] == 'agrees' for r in checked)
    hand = [r for r in rows if r.get('second_check', '').startswith('hand')]
    num['auditHand'], num['auditHandAgreed'] = len(hand), sum(r['second_check'] == 'hand: agrees' for r in hand)
    num['auditUnstatedByReference'] = sum(r['category'] == 'not_stated' and r['follows_benchmark'] == 'yes' for r in read)
    num['auditStreams'] = sum(r['category'] == 'strict_temporal' and ('online' in r['note'].lower() or 'stream' in r['note'].lower()) for r in read)


def main():
    logs_table()
    treatments()
    robustness()
    auc()
    methods()
    tuning()
    decisions()
    calibration_detail()
    anchor()
    prevalence()
    write('numbers.tex', ''.join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(num.items())))
    bad = [k for k in num if not k.isalpha()]
    if bad:
        raise SystemExit(f"macro names with characters TeX does not accept: {bad}")
    print(f"wrote {len(os.listdir(OUT))} files and {len(num)} numbers to {os.path.relpath(OUT)}")


if __name__ == '__main__':
    main()
