"""Parse BPI Challenge 2012 XES (untrusted download) into a compact pickle and print its shape."""
import gzip, pickle, sys
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
TERMINAL = {'A_APPROVED': 1, 'A_DECLINED': 2, 'A_CANCELLED': 3}


def local(tag):
    return tag.rsplit('}', 1)[-1]


cases, vocab = [], {}
with gzip.open(src, 'rb') as f:
    trace_attrs, events, in_event, ev = {}, [], False, {}
    for kind, el in ET.iterparse(f, events=('start', 'end')):
        tag = local(el.tag)
        if kind == 'start':
            if tag == 'trace':
                trace_attrs, events = {}, []
            elif tag == 'event':
                in_event, ev = True, {}
            continue
        if tag in ('string', 'date', 'int', 'float'):
            key, val = el.get('key'), el.get('value')
            (ev if in_event else trace_attrs)[key] = val
        elif tag == 'event':
            name = ev['concept:name']
            if name.startswith('W_'):
                name += '|' + ev.get('lifecycle:transition', '')
            ts = datetime.fromisoformat(ev['time:timestamp']).timestamp()
            events.append((vocab.setdefault(name, len(vocab) + 1), ts, ev['concept:name']))
            in_event = False
        elif tag == 'trace':
            events.sort(key=lambda e: e[1])
            acts = np.array([e[0] for e in events], dtype=np.int64)
            ts = np.array([e[1] for e in events], dtype=np.float64)
            term_idx, cause = -1, 0
            for i, e in enumerate(events):
                if e[2] in TERMINAL:
                    term_idx, cause = i, TERMINAL[e[2]]
                    break
            try:
                amount = float(trace_attrs.get('AMOUNT_REQ', 'nan'))
            except ValueError:
                amount = float('nan')
            cases.append(dict(acts=acts, ts=ts, term_idx=term_idx, cause=cause, amount=amount))
            el.clear()

with open(dst, 'wb') as f:
    pickle.dump(dict(cases=cases, vocab=vocab), f)

DAY = 86400.0
starts = np.array([c['ts'][0] for c in cases])
ends = np.array([c['ts'][-1] for c in cases])
cause = np.array([c['cause'] for c in cases])
dur = np.array([(c['ts'][c['term_idx']] - c['ts'][0]) / DAY if c['cause'] else np.nan for c in cases])
print('cases', len(cases), 'events', sum(len(c['acts']) for c in cases), 'vocab', len(vocab))
print('log start', datetime.fromtimestamp(starts.min()), 'last case start', datetime.fromtimestamp(starts.max()),
      'log end', datetime.fromtimestamp(ends.max()))
print('outcome counts [running, approved, declined, cancelled]:', np.bincount(cause, minlength=4).tolist())
for c, n in ((1, 'approved'), (2, 'declined'), (3, 'cancelled')):
    d = dur[cause == c]
    print(f'{n:9s} days-to-outcome quantiles 10/25/50/75/90/99:', np.round(np.quantile(d, [.1, .25, .5, .75, .9, .99]), 2).tolist())
d = dur[cause > 0]
for h in (1, 3, 7, 14, 30):
    print(f'P(outcome within {h:2d}d of case start | resolved) = {np.mean(d <= h):.3f}')
print('events before terminal: mean', np.mean([c['term_idx'] if c['cause'] else len(c['acts']) for c in cases]).round(1),
      'max', max(len(c['acts']) for c in cases))
