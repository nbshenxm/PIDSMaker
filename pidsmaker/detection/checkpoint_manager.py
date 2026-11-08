"""Checkpoint manager for saving and loading model weights at different epochs.

Handles:
- Saving model checkpoints with metadata
- Loading checkpoints from disk
- Tracking best checkpoint based on validation score
- Cleaning up old checkpoints if needed
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from pidsmaker.encoders import TGNEncoder
from pidsmaker.utils.utils import log


class CheckpointManager:
    """Manages model checkpoints for different epochs.
    
    Attributes:
        checkpoint_dir: Directory to store checkpoints
        keep_every_n: Only keep checkpoints every N epochs (1 = keep all)
        model: Reference to model for saving/loading
        cfg: Configuration object
    """
    
    def __init__(self, checkpoint_dir: str, keep_every_n: int = 1, cfg=None):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory path for storing checkpoints
            keep_every_n: Only keep checkpoints at intervals (1 = all epochs)
            cfg: Configuration object (needed for save/load operations)
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_every_n = keep_every_n
        self.cfg = cfg
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.metadata_file = os.path.join(checkpoint_dir, "checkpoints.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata from disk."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "checkpoints": {},
            "best_checkpoint": None,
            "best_val_score": float("-inf"),
        }
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def should_save_epoch(self, epoch: int) -> bool:
        """Check if checkpoint should be saved for this epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            bool: True if checkpoint should be saved
        """
        if self.keep_every_n == 1:
            return True
        return (epoch + 1) % self.keep_every_n == 0
    
    def get_checkpoint_path(self, epoch: int) -> str:
        """Get the checkpoint directory path for an epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            str: Path to checkpoint directory
        """
        return os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}")
    
    def save_checkpoint(
        self,
        model,
        epoch: int,
        val_score: Optional[float] = None,
        train_loss: Optional[float] = None,
    ) -> str:
        """Save model checkpoint for a specific epoch.
        
        Args:
            model: Model to save
            epoch: Current epoch number
            val_score: Validation score (for tracking best model)
            train_loss: Training loss value
            
        Returns:
            str: Path to saved checkpoint
        """
        if not self.should_save_epoch(epoch):
            return None
        
        checkpoint_path = self.get_checkpoint_path(epoch)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model state
        self._save_model_state(model, checkpoint_path)
        
        # Update metadata
        self.metadata["checkpoints"][str(epoch)] = {
            "epoch": epoch,
            "val_score": val_score,
            "train_loss": train_loss,
            "path": checkpoint_path,
        }
        
        # Update best checkpoint if applicable
        if val_score is not None and val_score > self.metadata["best_val_score"]:
            self.metadata["best_checkpoint"] = str(epoch)
            self.metadata["best_val_score"] = val_score
        
        self._save_metadata()

        val_score_str = f"{val_score:.4f}" if val_score is not None else "N/A"
        
        log(
            f"Checkpoint saved: epoch {epoch} | val_score: {val_score_str} | path: {checkpoint_path}",
            return_line=True,
        )
        
        return checkpoint_path
    
    def load_checkpoint(self, model, epoch: int):
        """Load a checkpoint into the model.
        
        Args:
            model: Model to load weights into
            epoch: Epoch number to load
            
        Returns:
            bool: True if successfully loaded
        """
        checkpoint_path = self.get_checkpoint_path(epoch)
        
        if not os.path.exists(checkpoint_path):
            log(f"Checkpoint not found: {checkpoint_path}", return_line=True)
            return False
        
        self._load_model_state(model, checkpoint_path)
        log(f"Checkpoint loaded: epoch {epoch} from {checkpoint_path}", return_line=True)
        return True
    
    def load_best_checkpoint(self, model):
        """Load the best checkpoint (based on validation score).
        
        Args:
            model: Model to load weights into
            
        Returns:
            int: Epoch number of loaded checkpoint, or -1 if not found
        """
        best_epoch = self.metadata.get("best_checkpoint")
        if best_epoch is None:
            log("No best checkpoint found in metadata", return_line=True)
            return -1
        
        best_epoch = int(best_epoch)
        success = self.load_checkpoint(model, best_epoch)
        return best_epoch if success else -1
    
    def _save_model_state(self, model, path: str):
        """Save model state dict and encoder-specific state."""
        os.makedirs(path, exist_ok=True)
        
        # Save model state dict
        torch.save(
            model.state_dict(),
            os.path.join(path, "state_dict.pkl"),
            pickle_protocol=4,
        )
        
        # Save encoder-specific state if using TGN
        if isinstance(model.encoder, TGNEncoder):
            torch.save(
                model.encoder.neighbor_loader,
                os.path.join(path, "neighbor_loader.pkl"),
                pickle_protocol=4,
            )
            if hasattr(model.encoder, 'memory') and model.encoder.memory is not None:
                torch.save(
                    model.encoder.memory,
                    os.path.join(path, "memory.pkl"),
                    pickle_protocol=4,
                )
    
    def _load_model_state(self, model, path: str):
        """Load model state dict and encoder-specific state."""
        state_dict_path = os.path.join(path, "state_dict.pkl")
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(f"State dict not found: {state_dict_path}")
        
        model.load_state_dict(torch.load(state_dict_path))
        
        # Load encoder-specific state if using TGN
        if isinstance(model.encoder, TGNEncoder):
            neighbor_loader_path = os.path.join(path, "neighbor_loader.pkl")
            if os.path.exists(neighbor_loader_path):
                model.encoder.neighbor_loader = torch.load(neighbor_loader_path)
            
            memory_path = os.path.join(path, "memory.pkl")
            if os.path.exists(memory_path):
                model.encoder.memory = torch.load(memory_path)
    
    def list_checkpoints(self) -> Dict[int, Dict[str, Any]]:
        """List all available checkpoints with their metadata.
        
        Returns:
            dict: Checkpoint metadata indexed by epoch
        """
        checkpoints = {}
        for epoch_str, metadata in self.metadata["checkpoints"].items():
            checkpoints[int(epoch_str)] = metadata
        return dict(sorted(checkpoints.items()))
    
    def get_checkpoint_info(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific checkpoint.
        
        Args:
            epoch: Epoch number
            
        Returns:
            dict: Checkpoint metadata or None if not found
        """
        return self.metadata["checkpoints"].get(str(epoch))
    
    def delete_checkpoint(self, epoch: int) -> bool:
        """Delete a checkpoint from disk and metadata.
        
        Args:
            epoch: Epoch number to delete
            
        Returns:
            bool: True if successfully deleted
        """
        checkpoint_path = self.get_checkpoint_path(epoch)
        
        try:
            import shutil
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            
            if str(epoch) in self.metadata["checkpoints"]:
                del self.metadata["checkpoints"][str(epoch)]
                self._save_metadata()
            
            log(f"Checkpoint deleted: epoch {epoch}", return_line=True)
            return True
        except Exception as e:
            log(f"Error deleting checkpoint {epoch}: {e}", return_line=True)
            return False
