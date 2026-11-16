"""Check the stored results and the manuscript's headline numbers without trusting the summaries.

    python censoring/verify.py                       re-run helpdesk and bpi12 from the raw logs, then recompute
    python censoring/verify.py --logs traffic,bpi20  re-run other logs (any of datasets.py)
    python censoring/verify.py --no-rerun            only the recomputation (seconds)

1. Re-run. study.py is run again for the given logs, into a temporary folder, and every number of every line is
   compared with the stored line in results/<log>.jsonl. Run times and package versions are left out.
2. Recompute. The headline numbers of the journal manuscript are computed again from the stored result files,
   with code that shares nothing with pooled.py, auc_summary.py or journal_inputs.py, and compared with
   paper_journal/generated/numbers.tex.

Nothing is written into results/ or into the manuscript. The exit status is 0 if everything agrees.
This is the check an author can run on their own machine before putting their name to the numbers.
"""
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
NUMBERS = os.path.join(HERE, '..', '..', 'paper_journal', 'generated', 'numbers.tex')
BASE = 'censor_aware'
SKIP = ('fit_s', 'seconds')   # run times


def flat(d, prefix=''):
    """Every number of a result line, with the path that leads to it."""
    for k, v in d.items():
        if k == 'versions':
            continue
        if isinstance(v, dict):
            yield from flat(v, prefix + k + '.')
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    yield f'{prefix}{k}[{i}]', x
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and not k.endswith(SKIP):
            yield prefix + k, v


def lines(path):
    return {(r['cutoff'], r['history_days']): r for r in map(json.loads, open(path))}


def rerun(logs):
    """-> number of numbers that differ from the stored ones."""
    wrong = 0
    with tempfile.TemporaryDirectory() as tmp:
        for log in logs:
            out = os.path.join(tmp, log + '.jsonl')
            print(f're-running {log} ...', flush=True)
            done = subprocess.run([sys.executable, os.path.join(HERE, 'study.py'), log, '--out', out], capture_output=True, text=True)
            if done.returncode:
                print(done.stderr[-2000:])
                return 1
            new, old = lines(out), lines(os.path.join(RESULTS, log + '.jsonl'))
            compared = differ = 0
            for key in old:
                if key not in new:
                    print(f'  {log} {key}: not produced by the re-run')
                    differ += 1
                    continue
                a, b = dict(flat(new[key])), dict(flat(old[key]))
                for name in sorted(a.keys() | b.keys()):
                    compared += 1
                    if name not in a or name not in b or a[name] != b[name]:
                        differ += 1
                        if differ <= 5:
                            print(f'  {log} {key} {name}: now {a.get(name)}, stored {b.get(name)}')
            print(f'  {log}: {len(old)} settings, {compared} numbers compared, {differ} differ')
            wrong += differ
    return wrong


def random_effects(y, se):
    """DerSimonian-Laird estimate of the mean over units."""
    v = [max(s * s, 1e-12) for s in se]
    w = [1 / x for x in v]
    fixed = sum(a * b for a, b in zip(w, y)) / sum(w)
    q = sum(a * (b - fixed) ** 2 for a, b in zip(w, y))
    tau2 = max(0.0, (q - (len(y) - 1)) / (sum(w) - sum(a * a for a in w) / sum(w)))
    w = [1 / (x + tau2) for x in v]
    return sum(a * b for a, b in zip(w, y)) / sum(w)


def recompute():
    """-> {macro name: value}, from the result files of the main model alone."""
    sys.path.insert(0, HERE)
    from datasets import DATASETS   # only for the number of deadlines of a log
    rows = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, '*.jsonl'))):
        log = os.path.basename(path)[:-len('.jsonl')]
        if log in DATASETS:
            rs = list(lines(path).values())
            lengths = sorted({r['history_days'] for r in rs})
            for r in rs:
                r['rank'] = lengths.index(r['history_days'])
            rows[log] = rs
    got = {'settings': sum(len(rs) for rs in rows.values())}
    for treat, name in (('complete_only', 'complete'), ('open_as_negative', 'negative'), ('mature_only', 'mature')):
        ci = [r['brier_minus_censor_aware'][treat]['ci95'] for rs in rows.values() for r in rs
              if (r.get('brier_minus_censor_aware') or {}).get(treat)]
        got[name + 'Worse'], got[name + 'Better'] = sum(lo > 0 for lo, _ in ci), sum(hi < 0 for _, hi in ci)
        for rank, letter in enumerate('ABCD'):
            y, se = [], []
            for rs in rows.values():
                v = [math.log(r[treat]['brier_ident'] / r[BASE]['brier_ident']) for r in rs if r['rank'] == rank and r.get(treat) and r.get(BASE)]
                if len(v) >= 2:
                    m = sum(v) / len(v)
                    y.append(m)
                    se.append(math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v)))
            if len(y) >= 2:
                got[f'{name}Pooled{letter}'] = 100 * (math.exp(random_effects(y, se)) - 1)
                got[f'{name}Plain{letter}'] = 100 * (math.exp(sum(y) / len(y)) - 1)
    # AUC and Brier score in points, complete cases only against the window rule, on the same settings:
    # those in which the middle deadline is shorter than the history and both have an AUC
    for rank, letter in enumerate('ABCD'):
        auc, brier = [], []
        for log, rs in rows.items():
            use = [r for r in rs if r['rank'] == rank and r.get('complete_only') and r.get(BASE)
                   and r['complete_only'].get('auc_any_mid') is not None and r[BASE].get('auc_any_mid') is not None
                   and len(r['identifiable_deadlines']) > len(DATASETS[log]['deadlines']) // 2]
            if use:
                auc.append(sum(r['complete_only']['auc_any_mid'] - r[BASE]['auc_any_mid'] for r in use) / len(use))
                brier.append(sum(r['complete_only']['brier_ident'] - r[BASE]['brier_ident'] for r in use) / len(use))
        got[f'aucComplete{letter}'] = abs(100 * sum(auc) / len(auc))
        got[f'brierComplete{letter}'] = abs(100 * sum(brier) / len(brier))
    return got


def compare(got):
    """-> number of macros of the manuscript that differ from the recomputed values."""
    macros = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', open(NUMBERS).read()))
    wrong = checked = 0
    for name, value in got.items():
        if name not in macros:
            continue
        text = macros[name].replace('{,}', '').replace(',', '')
        try:
            written = float(text)
        except ValueError:
            continue
        digits = len(text.split('.')[1]) if '.' in text else 0
        mine = round(value, digits)
        checked += 1
        if abs(abs(mine) - abs(written)) > 10 ** -digits / 2:
            wrong += 1
            print(f'  \\{name}: manuscript {macros[name]}, recomputed {mine:.{digits}f}')
    print(f'  {checked} headline numbers recomputed from the result files, {wrong} differ from the manuscript')
    return wrong


def main():
    argv = sys.argv[1:]
    logs = argv[argv.index('--logs') + 1].split(',') if '--logs' in argv else ['helpdesk', 'bpi12']
    wrong = 0 if '--no-rerun' in argv else rerun(logs)
    print('recomputing the headline numbers ...')
    wrong += compare(recompute())
    print('all agree' if not wrong else f'{wrong} DIFFERENCES')
    sys.exit(1 if wrong else 0)


if __name__ == '__main__':
    main()
