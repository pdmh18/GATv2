"""Causal diagnostic rules; no model training or scenario/profile oracle features."""
from __future__ import annotations
import bisect
from collections import Counter, defaultdict, deque


class CausalRules:
    def __init__(self):
        self.amounts = defaultdict(list)
        self.by_type = defaultdict(list)
        self.hours = defaultdict(Counter)
        self.channels = defaultdict(Counter)
        self.receivers = defaultdict(set)
        self.recent = defaultdict(deque)
        self.incoming = defaultdict(deque)

    def score_then_observe(self, row, at):
        src, dst, kind = row['src_acc'], row['dst_acc'], row['type']
        history, typed = self.amounts[src], self.by_type[(src, kind)]
        n = len(history)
        chosen = typed if len(typed) >= 5 else history
        mid = len(chosen) // 2
        median = chosen[mid] if len(chosen) % 2 else (chosen[mid-1] + chosen[mid]) / 2 if chosen else 0
        ratio = row['amount'] / median if median and n >= 20 else 0
        hour = int(row['timestamp'][11:13])
        uncommon_hour = n >= 30 and self.hours[src][hour] / n < .01
        uncommon_channel = n >= 30 and self.channels[src][row['channel']] / n < .15
        new_receiver = dst not in self.receivers[src]
        recent = self.recent[src]
        while recent and recent[0] < at - 120:
            recent.popleft()
        for aid in (src, dst):
            queue = self.incoming[aid]
            while queue and queue[0][0] < at - 600:
                queue.popleft()
        context = min(ratio, 12) * (uncommon_hour + uncommon_channel + new_receiver) / 3
        velocity = len(recent) / 2
        graph = 0
        if kind == 'TRANSFER':
            graph = max(len({s for _, s in self.incoming[dst]}), len({s for _, s in self.incoming[src]})) / 3
        scores = {'amount_only': ratio, 'behavior': max(context, velocity),
                  'behavior_graph': max(context, velocity, graph)}
        # Every score is computed before observing the current request, without label.
        bisect.insort(history, row['amount'])
        bisect.insort(typed, row['amount'])
        self.hours[src][hour] += 1
        self.channels[src][row['channel']] += 1
        self.receivers[src].add(dst)
        recent.append(at)
        if kind == 'TRANSFER':
            self.incoming[dst].append((at, src))
        return scores


def binary_metrics(rows, threshold):
    tp = fp = fn = tn = 0
    for score, label, _ in rows:
        pred = score >= threshold
        tp += pred and label == 1
        fp += pred and label == 0
        fn += not pred and label == 1
        tn += not pred and label == 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0)


def ranking_metrics(rows):
    ordered = sorted(rows, key=lambda r: r[0])
    positives = sum(r[1] for r in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError('Baseline evaluation requires both classes')
    seen_negative = wins = 0
    groups = []
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        pos = sum(r[1] for r in ordered[i:j]); neg = j - i - pos
        wins += pos * (seen_negative + neg / 2)
        seen_negative += neg
        groups.append((pos, neg))
        i = j
    tp = fp = ap = 0
    for pos, neg in reversed(groups):
        tp += pos; fp += neg
        ap += pos / positives * tp / (tp + fp)
    return dict(roc_auc=wins/(positives*negatives), average_precision=ap)
