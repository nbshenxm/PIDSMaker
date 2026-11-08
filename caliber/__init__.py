"""CALIBER: a calibrated alert / defer / dismiss decision layer over PIDS scores.

Design rule: only `caliber.pidsmaker_bridge` imports PIDSMaker. Everything else
works on the files PIDSMaker leaves behind (per-epoch edge-loss CSVs) plus the
manifests and label exports written by the bridge, so the decision layer can be
iterated on offline, on CPU, without retraining any detector.
"""
