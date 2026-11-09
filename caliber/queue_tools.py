"""Queue maintenance for scripts/caliber/{enqueue.sh,worker.sbatch}.

  python -m caliber.queue_tools reconcile --artifact_dir A [--queue Q] [--requeue_failed]

reconcile: for every task in pending/ (and failed/ with --requeue_failed), compute
its PIDSMaker task paths; if training AND evaluation are already done on disk,
write its manifest, pack its edge losses, and move the task to done/ instead of
retraining. Otherwise (re)place it in pending/.
"""

import argparse
import os
import shlex

from caliber.pidsmaker_bridge import cmd_manifest, load_cfg

DEFAULT_QUEUE = "/fs1/projects/benchmarking_PIDS/sherwin_dir/caliber_queue"


def read_task(path):
    kv = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                kv[k] = v
    return kv


def task_argv(t, artifact_dir):
    return [t["SYSTEM"], t["DATASET"], "--artifact_dir", artifact_dir,
            "--training.seed", t["SEED"], *shlex.split(t.get("ARGS", ""))]


def is_complete(argv):
    cfg = load_cfg(argv)
    return (os.path.exists(os.path.join(cfg.training._task_path, "done.txt"))
            and os.path.exists(os.path.join(cfg.evaluation._task_path, "done.txt")))


def reconcile(artifact_dir, queue, requeue_failed=False):
    os.environ["CALIBER_PACK"] = "1"
    srcs = ["pending"] + (["failed"] if requeue_failed else [])
    moved_done = moved_pending = 0
    for src in srcs:
        d = os.path.join(queue, src)
        for name in sorted(os.listdir(d)):
            if name.startswith("."):
                continue
            path = os.path.join(d, name)
            try:
                t = read_task(path)
                argv = task_argv(t, artifact_dir)
                if is_complete(argv):
                    cmd_manifest(argv)
                    os.replace(path, os.path.join(queue, "done", name))
                    moved_done += 1
                elif src != "pending":
                    os.replace(path, os.path.join(queue, "pending", name))
                    moved_pending += 1
            except FileNotFoundError:
                continue                     # a worker claimed it meanwhile
    print(f"reconcile: {moved_done} already complete -> done/, {moved_pending} requeued -> pending/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["reconcile"])
    ap.add_argument("--artifact_dir", required=True)
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--requeue_failed", action="store_true")
    a = ap.parse_args()
    reconcile(a.artifact_dir, a.queue, a.requeue_failed)


if __name__ == "__main__":
    main()
