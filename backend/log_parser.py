"""
Reads logs/train_log.txt (written by the NAS training pipeline) and
returns structured JSON for the frontend charts.

Log line format examples:
  PRETRAIN | epoch=5 | loss=2.34 | acc=0.21 | lr=0.024
  SEARCH   | episode=10 | accuracy=0.83 | flops=48200000 | raw_reward=0.72 | reward=1.23 | entropy=1.89 | is_new=True
"""

import re
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'train_log.txt')


def parse_log():
    """Parse train_log.txt and return pretrain + search data."""
    pretrain_epochs = []
    search_episodes = []

    if not os.path.exists(LOG_PATH):
        return pretrain_epochs, search_episodes

    # Pretrain epoch line: matches "     1    3.6232      10.43%    0.024975"
    pretrain_pattern = re.compile(
        r'^\s*(?P<epoch>\d+)\s+(?P<loss>[\d\.]+)\s+(?P<acc>[\d\.]+)%\s+(?P<lr>[\d\.]+)$'
    )

    # Search episode line: matches "    10   13.32%     88.5M   0.1280  -1.8001   2.079   yes   21.4s"
    search_pattern = re.compile(
        r'^\s*(?P<episode>\d+)\s+(?P<acc>[\d\.]+)%\s+(?P<flops>[\d\.]+)M\s+(?P<raw_reward>[\-\d\.]+)\s+(?P<reward>[\-\d\.]+)\s+(?P<entropy>[\d\.]+)\s+(?P<new>yes|no)\s+(?P<time>[\d\.]+)s$'
    )

    with open(LOG_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Try to match pretrain line
            pm = pretrain_pattern.match(line)
            if pm:
                try:
                    pretrain_epochs.append({
                        'epoch':    int(pm.group('epoch')),
                        'loss':     float(pm.group('loss')),
                        'accuracy': float(pm.group('acc')) / 100.0,
                        'lr':       float(pm.group('lr')),
                    })
                except ValueError:
                    pass
                continue

            # Try to match search line
            sm = search_pattern.match(line)
            if sm:
                try:
                    search_episodes.append({
                        'episode':    int(sm.group('episode')),
                        'accuracy':   float(sm.group('acc')) / 100.0,
                        'flops':      float(sm.group('flops')) * 1e6,
                        'raw_reward': float(sm.group('raw_reward')),
                        'reward':     float(sm.group('reward')),
                        'entropy':    float(sm.group('entropy')),
                        'is_new':     sm.group('new') == 'yes',
                    })
                except ValueError:
                    pass
                continue

    return pretrain_epochs, search_episodes


def get_status():
    """Determine pipeline phase from log content."""
    if not os.path.exists(LOG_PATH):
        return 'idle'

    pretrain, search = parse_log()

    # Check if exports exist (done)
    export_path = os.path.join(os.path.dirname(__file__), '..', 'exports', 'search_results.json')
    smoke_export = os.path.join(os.path.dirname(__file__), '..', 'exports', 'smoke', 'search_results.json')
    if os.path.exists(export_path) or os.path.exists(smoke_export):
        return 'complete'
    if search:
        return 'searching'
    if pretrain:
        return 'pretraining'
    return 'idle'
