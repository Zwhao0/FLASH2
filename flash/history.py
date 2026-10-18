"""Bounded, iteration-indexed consumption of the paper's rolling history."""
import json

HISTORY_LIMIT = 500

def advance_history(history, raw_rows, last_seen, limit=HISTORY_LIMIT):
    if limit <= 0:
        raise ValueError('history limit must be positive')
    fresh = {}
    for raw in raw_rows:
        row = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        iteration = int(row['iteration'])
        if iteration > last_seen:
            fresh[iteration] = row
    new = [fresh[index] for index in sorted(fresh)]
    gap = max(0, int(new[0]['iteration']) - last_seen - 1) if new else 0
    return (history + new)[-limit:], new, int(new[-1]['iteration']) if new else last_seen, gap
