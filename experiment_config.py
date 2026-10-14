"""Predeclared protocol and output protection; no wet-dependent settings."""
from pathlib import Path
from dataset_lstm import WINDOW_CONFIG

DEFAULT_RESULTS = 'results_onset_v2'
PROTOCOL_ID = 'onset_within_250ms_v2'
MODEL_KINDS = ['lstm', 'rnn', 'last_frame']

def writable_output(path):
    resolved = Path(path).resolve()
    archive = (Path(__file__).resolve().parent / 'reference_results').resolve()
    if resolved == archive or archive in resolved.parents:
        raise ValueError('reference_results is immutable; choose a new working output directory')
    return resolved

def window_config(manifest):
    if 'window_config' in manifest:
        return manifest['window_config']
    # The untouched 2026-10-14 archive used confirmation entirely in horizon.
    return {**WINDOW_CONFIG, 'confirmation_policy': 'within_horizon'}

def make_model(kind):
    from model_lstm import HandlingLSTM
    from model_rnn import HandlingRNN
    from model_last_frame import LastFrameMLP
    factories = {'lstm': HandlingLSTM, 'rnn': HandlingRNN, 'last_frame': LastFrameMLP}
    return factories[kind]()
