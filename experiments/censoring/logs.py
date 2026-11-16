"""Read an event log (XES, gzipped or not, or a CSV) as a stream of cases.

Every reader yields (case_attributes, events) with events = [(activity, lifecycle, unix_time, event_attributes)] sorted by
time; events with equal times keep their order in the file. The files are untrusted downloads: nothing in them is
executed, and the scripts that call this run with `python -I`.
"""
import csv
import gzip
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

VALUE_TAGS = ('string', 'date', 'int', 'float', 'boolean', 'id')


def _open(path):
    return gzip.open(path, 'rb') if str(path).endswith('.gz') else open(path, 'rb')


def _time(text):
    t = datetime.fromisoformat(text)
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


def read_xes(path):
    with _open(path) as f:
        case, events, ev = {}, [], None
        for kind, el in ET.iterparse(f, events=('start', 'end')):
            tag = el.tag.rsplit('}', 1)[-1]
            if kind == 'start':
                if tag == 'trace':
                    case, events = {}, []
                elif tag == 'event':
                    ev = {}
                continue
            if tag in VALUE_TAGS:
                (case if ev is None else ev)[el.get('key')] = el.get('value')
            elif tag == 'event':
                events.append((ev.get('concept:name', ''), ev.get('lifecycle:transition', ''), _time(ev['time:timestamp']), ev))
                ev = None
            elif tag == 'trace':
                events.sort(key=lambda e: e[2])
                yield case, events
                el.clear()


def read_csv(path, case='Case ID', activity='Activity', time='Complete Timestamp', time_format='%Y/%m/%d %H:%M:%S.%f',
             delimiter=','):
    """A CSV with one event per row. The first row of a case supplies the case attributes."""
    cases = {}
    with open(path, newline='', encoding='utf-8-sig', errors='replace') as f:
        rows = csv.reader(f, delimiter=delimiter)
        header = next(rows)
        col = {name: header.index(name) for name in (case, activity, time)}  # first column of that name
        for row in rows:
            attrs = dict(zip(header, row))
            t = datetime.strptime(row[col[time]], time_format).replace(tzinfo=timezone.utc).timestamp()
            cases.setdefault(row[col[case]], []).append((row[col[activity]], '', t, attrs))
    for cid, events in cases.items():
        events.sort(key=lambda e: e[2])
        yield {'concept:name': cid, **events[0][3]}, events


def read_log(path, **csv_columns):
    return read_csv(path, **csv_columns) if str(path).endswith('.csv') else read_xes(path)
