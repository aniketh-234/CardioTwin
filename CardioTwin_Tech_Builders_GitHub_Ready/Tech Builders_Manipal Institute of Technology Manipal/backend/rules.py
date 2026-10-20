"""Transparent sandbox trend rules; thresholds are engineering assumptions."""
import numpy as np

RULE_VERSION = 'sandbox-trends-v1'
# Compare recent seven-day means with the preceding 21 days. A training
# standard deviation provides a stable reference without dividing by tiny
# within-patient variances. These thresholds are not clinical cutoffs.
SIGNALS = [(0, 'resting_hr', 1), (1, 'hrv_rmssd', -1),
           (2, 'steps', -1), (3, 'sleep_hours', -1),
           (4, 'home_systolic_bp', 1)]


def trend_rules(sequence, scales):
    checks = []
    for k, name, direction in SIGNALS:
        old, recent = sequence[:21, k], sequence[21:, k]
        n_old, n_recent = int(np.isfinite(old).sum()), int(np.isfinite(recent).sum())
        check = {'signal': name, 'baseline_count': n_old, 'recent_count': n_recent,
                 'direction': 'increase' if direction == 1 else 'decrease',
                 'threshold_sd': .5, 'triggered': False, 'available': False}
        if n_old >= 14 and n_recent >= 5:
            baseline, current = float(np.nanmean(old)), float(np.nanmean(recent))
            change = current - baseline
            standardized = direction * change / max(float(scales[k]), 1e-6)
            check.update(available=True, baseline=baseline, recent=current,
                         change=change, standardized_change=standardized,
                         triggered=bool(standardized >= .5))
        checks.append(check)
    triggered = [c['signal'] for c in checks if c['triggered']]
    available = sum(c['available'] for c in checks)
    return {'version': RULE_VERSION, 'checks': checks, 'triggered_signals': triggered,
            'review_flag': len(triggered) >= 2, 'available_signals': available,
            'minimum_triggered_signals': 2,
            'note': 'Synthetic trend heuristic: at least two directional changes of 0.5 training standard deviations. Not a clinical alert.'}
