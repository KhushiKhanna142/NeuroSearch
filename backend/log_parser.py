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

    with open(LOG_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('PRETRAIN'):
                m = {k: v for k, v in re.findall(r'(\w+)=([^\s|]+)', line)}
                try:
                    pretrain_epochs.append({
                        'epoch':    int(m.get('epoch', 0)),
                        'loss':     float(m.get('loss', 0)),
                        'accuracy': float(m.get('acc', 0)),
                        'lr':       float(m.get('lr', 0)),
                    })
                except (ValueError, KeyError):
                    pass

            elif line.startswith('SEARCH'):
                m = {k: v for k, v in re.findall(r'(\w+)=([^\s|]+)', line)}
                try:
                    search_episodes.append({
                        'episode':    int(m.get('episode', 0)),
                        'accuracy':   float(m.get('accuracy', 0)),
                        'flops':      float(m.get('flops', 0)),
                        'raw_reward': float(m.get('raw_reward', 0)),
                        'reward':     float(m.get('reward', 0)),
                        'entropy':    float(m.get('entropy', 0)),
                        'is_new':     m.get('is_new', 'True') == 'True',
                    })
                except (ValueError, KeyError):
                    pass

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
