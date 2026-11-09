"""Stage frozen member weights and enqueue realizable-attack scoring runs.

For each member of a weight-saving group (runs trained with --save_weights), copy the
selected epoch's weights (min benign validation loss, the same rule CALIBER uses) and the
frozen word2vec model into
  /fs1/.../sherwin_dir/caliber_weights/<system>/<dataset>/seed<k>/{state_dict.pkl,word2vec.model}
and enqueue one task per (attack, member):
  pidsmaker main ... --from_weights --weights_dir <dir> --adversary <json> --training.seed <k>
Clean scores for the same members are the original runs (identical weights and epoch).

  python -m caliber.adv_stage --artifact_dir A --system velox --dataset CADETS_E3 --group G
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess

from caliber import data

ATTACKS = [{"rename": True}, {"borrow": True}, {"substitute": True}, {"dilute": 5},
           {"prime": 2}, {"rename": True, "borrow": True, "dilute": 5}]
WEIGHTS_ROOT = "/fs1/projects/benchmarking_PIDS/sherwin_dir/caliber_weights"


def stage(m, cache):
    ep = data.select_epoch(m, "min_val_loss", cache)
    src = os.path.join(m["training_task_path"], "trained_models", f"model_epoch_{ep}", "state_dict.pkl")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    # word2vec of this run's featurization task (the training task path's sibling chain)
    root = m["training_task_path"].split("/training/")[0]
    w2v = sorted(glob.glob(os.path.join(root, "featurization", m["dataset"], "featurization", "*",
                                        "stored_models", "word2vec.model")),
                 key=os.path.getmtime)
    if not w2v:
        raise FileNotFoundError("word2vec.model for " + m["dataset"])
    dst = os.path.join(WEIGHTS_ROOT, m["system"], m["dataset"], f"seed{m['seed']}")
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(src, os.path.join(dst, "state_dict.pkl"))
    for f in glob.glob(w2v[-1] + "*"):                 # word2vec.model (+ .npy side files)
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    return dst, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", required=True)
    args = ap.parse_args()
    cache = os.path.join(args.artifact_dir, "caliber", "cache")
    members = data.group_manifests(data.load_manifests(
        args.artifact_dir, args.system, args.dataset, args.group))[(args.system, args.dataset, args.group)]
    staged = []
    for m in members:
        d, ep = stage(m, cache)
        staged.append((m["seed"], d))
        print(f"staged seed {m['seed']} epoch {ep} -> {d}")
    for atk in ATTACKS:
        spec = json.dumps(atk, separators=(",", ":"))
        prev = ""
        for seed, d in sorted(staged):
            env = dict(os.environ, BASE_SEED=str(seed), REQUIRES=prev)
            subprocess.run(["bash", "scripts/caliber/enqueue.sh", args.system, args.dataset, "1",
                            "--from_weights", "--weights_dir", d, "--adversary", spec],
                           env=env, check=True)
            # members after the first wait for its construction (shared by all members);
            # task name hash = sha1 of the space-joined args, as in enqueue.sh
            if not prev:
                ah = hashlib.sha1(f"--from_weights --weights_dir {d} --adversary {spec}".encode()).hexdigest()[:8]
                prev = f"{args.system}__{args.dataset}__{ah}__seed{seed}"


if __name__ == "__main__":
    main()
