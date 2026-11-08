"""Velox trainer entrypoint.

This module now provides a concrete, usable Velox implementation built on the
shared batching, training, and evaluation infrastructure.
"""

import copy
import os
import tracemalloc

import torch

from pidsmaker.detection.checkpoint_manager import CheckpointManager
from pidsmaker.detection.training_methods import inference_loop
from pidsmaker.pids.velox.model import VeloxModel
from pidsmaker.pids.velox.score_artifacts import write_epoch_score_artifacts
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.factory import optimizer_factory
from pidsmaker.utils.utils import get_device, log, log_start, log_tqdm, set_seed


def remove_attacks_if_needed(graph, cfg):
    """Remove attack edges from graph for self-supervised training if configured."""
    if not cfg.training.decoder.few_shot.include_attacks_in_ssl_training:
        if 1 in graph.y:
            return graph.clone()[graph.y != 1]
    return graph


class VeloxTrainer:
    """A concrete trainer for Velox that owns its model construction."""

    @staticmethod
    def run(cfg):
        set_seed(cfg, seed=cfg.training.seed)
        log_start(__file__)
        device = get_device(cfg)

        if device == torch.device("cuda"):
            torch.cuda.reset_peak_memory_stats(device=device)
        tracemalloc.start()

        train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)

        if not train_data:
            raise ValueError("No training data was produced for Velox")

        sample = train_data[0][0]
        in_dim = sample.x_src.shape[1]
        out_dim = cfg.dataset.num_edge_types
        model = VeloxModel(cfg=cfg, in_dim=in_dim, out_dim=out_dim, device=device).to(device)
        optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))

        checkpoint_manager = None
        save_checkpoints = getattr(cfg.training_loop, "save_checkpoints", False)
        if save_checkpoints:
            checkpoints_dir = os.path.join(cfg.training._task_path, "checkpoints")
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoints_dir,
                keep_every_n=getattr(cfg.training_loop, "checkpoint_keep_every_n_epochs", 1),
                cfg=cfg,
            )

        best_selection_metric = float("-inf")
        best_state = None

        for epoch in range(cfg.training.num_epochs):
            model.train()
            model.reset_state()
            loss_acc = torch.zeros(1, device=device)
            total_loss = 0.0

            for dataset in train_data:
                for g in log_tqdm(dataset, "Velox training"):
                    g.to(device=device)
                    g = remove_attacks_if_needed(g, cfg)
                    optimizer.zero_grad()

                    results = model(g)
                    loss = results["loss"]
                    loss_acc += loss
                    total_loss += loss.item()

                    if (epoch + 1) % 1 == 0:
                        loss_acc.backward()
                        optimizer.step()
                        loss_acc = torch.zeros(1, device=device)

                    g.to("cpu")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            if loss_acc > 0:
                loss_acc.backward()
                optimizer.step()

            eval_stats = inference_loop.main(
                cfg=cfg,
                model=model,
                val_data=val_data,
                test_data=test_data,
                epoch=epoch,
                split="all",
                logging=False,
            )
            val_loss = eval_stats.get("val_loss")
            test_loss = eval_stats.get("test_loss")
            selection_metric = -float(val_loss)

            write_epoch_score_artifacts(
                cfg=cfg,
                model=model,
                val_data=val_data,
                test_data=test_data,
                epoch=epoch,
                device=device,
            )

            log(
                f"[@epoch{epoch:02d}] Velox eval - val_loss={val_loss:.4f} "
                f"| test_loss={test_loss:.4f} | selection_metric(-val_loss)={selection_metric:.6f}",
                return_line=True,
            )

            if checkpoint_manager is not None:
                checkpoint_manager.save_checkpoint(
                    model=model,
                    epoch=epoch,
                    val_score=selection_metric,
                    train_loss=total_loss,
                )

            if selection_metric > best_selection_metric:
                best_selection_metric = selection_metric
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        return best_selection_metric
