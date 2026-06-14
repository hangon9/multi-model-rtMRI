"""
Logging utilities for training.
Provides structured logging to both console and files.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import yaml
import shutil


class TrainingLogger:
    """
    Unified logger for training that logs to both console and files.
    
    Creates:
    - training.log: Overall training log with all messages
    - metrics.jsonl: Metrics in JSON Lines format for easy parsing
    - config.yaml: Copy of the configuration used for this run
    """
    
    def __init__(self, log_dir: str, config: dict, config_path: str = None):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            config: Training configuration dictionary
            run_name: Optional name for this run. If None, uses timestamp
            config_path: Optional path to original config file. If provided, copies the original file to preserve formatting
        """
        # Create run-specific directory with timestamp
        run_name = config.get("experiment_name", "baseline")+"_" + datetime.now().strftime("%Y%m%d_%H%M%S") 
        
        # Create log directory
        self.log_dir = Path(log_dir)
        self.run_dir = self.log_dir / run_name

        # If run directory already exists, remove it to start fresh
        if self.run_dir.exists():
            print(f"⚠️  Run directory '{run_name}' already exists. Removing old logs...")
            shutil.rmtree(self.run_dir)
        
        # Create fresh run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config - copy original file if available to preserve formatting
        saved_config_path = self.run_dir / "config.yaml"
        if config_path and Path(config_path).exists():
            shutil.copy2(config_path, saved_config_path)
        else:
            # Fallback to dumping dict with better formatting
            with open(saved_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Metrics file
        self.metrics_file = self.run_dir / "metrics.jsonl"
        
        self.logger.info(f"Logging initialized. Run directory: {self.run_dir}")
        self.logger.info(f"Config saved to: {saved_config_path}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with both file and console handlers."""
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        log_file = self.run_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, epoch: int, phase: str, metrics: dict, lr: float = None):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            phase: 'train' or 'val'
            metrics: Dictionary of metric names and values
            lr: Optional learning rate
        """
        # Log to console
        self.info(f"Epoch {epoch+1} - {phase.upper()}")
        self.info(f"  Loss: {metrics.get('loss_total', 0):.4f}")
        
        # Log loss components
        if 'loss_ce' in metrics or 'loss_l1' in metrics or 'loss_giou' in metrics:
            self.info("  Loss Components:")
            if 'loss_ce' in metrics:
                self.info(f"    loss_ce:    {metrics['loss_ce']:.4f}")
            if 'loss_cos' in metrics:
                self.info(f"    loss_cos:   {metrics['loss_cos']:.4f}")
            if 'loss_infonce' in metrics:
                self.info(f"    loss_infonce:  {metrics['loss_infonce']:.4f}")
        
        # Log prediction stats
        if any(k.startswith('pred_') for k in metrics):
            self.info("  Prediction Stats:")
            for k, v in metrics.items():
                if k.startswith('pred_'):
                    self.info(f"    {k}: {v:.4f}")
        
        # Log training stats
        if lr is not None or 'grad_norm' in metrics:
            self.info("  Training Stats:")
            if 'grad_norm' in metrics:
                self.info(f"    grad_norm:  {metrics['grad_norm']:.4f}")
            if lr is not None:
                self.info(f"    lr:         {lr:.6f}")
        
        # Save to metrics file in JSON Lines format
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch + 1,
            'phase': phase,
            **metrics
        }
        if lr is not None:
            metrics_entry['lr'] = lr
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')
    
    def log_model_info(self, model):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info("=" * 60)
        self.info("Model Information")
        self.info("=" * 60)
        self.info(f"Total parameters: {total_params:,}")
        self.info(f"Trainable parameters: {trainable_params:,}")
        self.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        self.info("=" * 60)
    
    def log_dataset_info(self, train_loader, val_loader=None):
        """Log dataset information."""
        self.info("=" * 60)
        self.info("Dataset Information")
        self.info("=" * 60)
        self.info(f"Training batches: {len(train_loader)}")
        if val_loader:
            self.info(f"Validation batches: {len(val_loader)}")
        self.info("=" * 60)
    
    def log_checkpoint(self, checkpoint_path: str, epoch: int, is_best: bool = False):
        """Log checkpoint save event."""
        checkpoint_type = "BEST" if is_best else "CHECKPOINT"
        self.info(f"{checkpoint_type} saved at epoch {epoch+1}: {checkpoint_path}")


def load_metrics(log_dir: Path, run_name: str) -> list:
    """
    Load metrics from a training run.
    
    Args:
        log_dir: Base log directory
        run_name: Name of the run
    
    Returns:
        List of metric dictionaries
    """
    metrics_file = log_dir / run_name / "metrics.jsonl"
    
    if not metrics_file.exists():
        return []
    
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    
    return metrics


def list_runs(log_dir: Path) -> list:
    """
    List all training runs.
    
    Args:
        log_dir: Base log directory
    
    Returns:
        List of run names (directory names)
    """
    if not log_dir.exists():
        return []
    
    runs = [d.name for d in log_dir.iterdir() if d.is_dir()]
    return sorted(runs, reverse=True)  # Most recent first
