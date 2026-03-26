"""
: SegFormer
NitroGen: AdamW + 

:
    - Batch size: 256
    - Learning rate: 0.0001
    - Weight decay: 0.1
    - : 8M
"""

import os
import sys

# [FIRE] 
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
import gc

# 
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# 
try:
    from models import NitroGenActionParser, ActionParsingLoss
    from dataset import FramePairDataset, create_train_val_dataloaders
except Exception as e:
    print(f"[ERROR] Failed to import modules: {e}")
    print("\nPlease ensure:")
    print("  1. transformers is installed: pip install transformers")
    print("  2. torch is installed: pip install torch torchvision")
    print("  3. All files are in the correct directory")
    sys.exit(1)

# 
DATA_DIR = "synthetic_data/frames"
TRAIN_ANNOTATIONS = "synthetic_data/annotations.json"
VAL_ANNOTATIONS = "synthetic_data/val_annotations.json"
OUTPUT_DIR = "checkpoints"
LOG_DIR = "logs"

# 
BATCH_SIZE = 4  # GPU32
NUM_EPOCHS = 1  # [FIRE] 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.1
IMAGE_SIZE = 256
NUM_WORKERS = 0  # Windows0
GRADIENT_ACCUMULATION_STEPS = 8  # batch size = 4 * 8 = 32
SAVE_EVERY = 1000


class Trainer:
    """
    
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: ActionParsingLoss,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        device: torch.device,
        output_dir: str,
        log_dir: str,
        use_amp: bool = False,
        memory_clean_interval: int = 0,
        early_stopping_patience: int = 10,
        min_delta: float = 0.001
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.use_amp = use_amp and AMP_AVAILABLE
        self.memory_clean_interval = memory_clean_interval
        self._num_epochs = 0  # train()

        # [FIRE] 
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.wait = 0  # 
        self.best_val_loss = float('inf')

        # 
        self.scaler = GradScaler() if self.use_amp else None

        # 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # 
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []

        # 
        if self.use_amp:
            print(f"  [FIRE] : ")
        if self.memory_clean_interval > 0:
            print(f"  [CLEAN] :  {self.memory_clean_interval} batch")
        if self.early_stopping_patience > 0:
            print(f"  [STOP] : patience={self.early_stopping_patience}, min_delta={self.min_delta}")

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """epoch"""
        self.model.train()

        epoch_losses = {
            "total": [],
            "joystick": [],
            "button": []
        }

        total_batches = len(train_loader)
        print_interval = max(1, total_batches // 100)  # 100
        start_time = datetime.now()

        for batch_idx, batch in enumerate(train_loader):
            # 
            frame_pairs = batch["frame_pair"].to(self.device)
            left_joystick = batch["left_joystick"].to(self.device)
            right_joystick = batch["right_joystick"].to(self.device)
            buttons = batch["buttons"].to(self.device)

            # 
            if self.use_amp:
                with autocast():
                    outputs = self.model(frame_pairs)
                    targets = {
                        "left_joystick": left_joystick,
                        "right_joystick": right_joystick,
                        "buttons": buttons
                    }
                    loss, losses = self.criterion(outputs, targets)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                # 
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(frame_pairs)
                targets = {
                    "left_joystick": left_joystick,
                    "right_joystick": right_joystick,
                    "buttons": buttons
                }
                loss, losses = self.criterion(outputs, targets)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

            # 
            for k, v in losses.items():
                epoch_losses[k].append(v)

            # 
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if self.use_amp:
                    # 
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # TensorBoard
                if self.global_step % 100 == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/lr", current_lr, self.global_step)

                # 
                if self.global_step % SAVE_EVERY == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # Nbatch
            if (batch_idx + 1) % max(1, print_interval // GRADIENT_ACCUMULATION_STEPS) == 0:
                progress = (batch_idx + 1) / total_batches * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                print(f"\rEpoch {epoch}/{self._num_epochs} | {progress:5.1f}% | "
                      f"{batch_idx+1:4d}/{total_batches} | loss={losses['total']:.4f} | "
                      f"ETA: {int(eta//60)}:{int(eta%60):02d}", end="", flush=True)

            # [CLEAN] 
            if self.memory_clean_interval > 0 and (batch_idx + 1) % self.memory_clean_interval == 0:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # batch
            del frame_pairs, left_joystick, right_joystick, buttons, outputs, targets, loss

        # 
        print()

        # epoch
        epoch_avg = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        return epoch_avg

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """"""
        self.model.eval()

        val_losses = []

        for batch in val_loader:
            # 
            frame_pairs = batch["frame_pair"].to(self.device)
            left_joystick = batch["left_joystick"].to(self.device)
            right_joystick = batch["right_joystick"].to(self.device)
            buttons = batch["buttons"].to(self.device)

            # 
            if self.use_amp:
                with autocast():
                    outputs = self.model(frame_pairs)
                    targets = {
                        "left_joystick": left_joystick,
                        "right_joystick": right_joystick,
                        "buttons": buttons
                    }
                    loss, _ = self.criterion(outputs, targets)
            else:
                outputs = self.model(frame_pairs)
                targets = {
                    "left_joystick": left_joystick,
                    "right_joystick": right_joystick,
                    "buttons": buttons
                }
                loss, _ = self.criterion(outputs, targets)

            val_losses.append(loss.item())

            # batch
            del frame_pairs, left_joystick, right_joystick, buttons, outputs, targets, loss

        avg_loss = sum(val_losses) / len(val_losses)
        return avg_loss

    def save_checkpoint(self, filename: str):
        """epoch checkpoint"""
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        # 

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int
    ):
        """"""
        self._num_epochs = num_epochs

        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        for epoch in range(1, num_epochs + 1):
            # 
            train_losses = self.train_epoch(train_loader, epoch)
            print()  # 

            # 
            val_loss = self.validate(val_loader)

            # 
            self.train_losses.append(train_losses)
            self.val_losses.append(val_loss)

            # [FIRE] 
            if val_loss < self.best_val_loss - self.min_delta:
                # 
                self.best_val_loss = val_loss
                self.wait = 0  # 

                # 
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                }, self.output_dir / "best_model.pt")
                print(f"  [OK] New best model! (val_loss: {val_loss:.4f})")
            else:
                # 
                self.wait += 1
                print(f"  [WAIT] No improvement for {self.wait} epoch(s)")

            # TensorBoard
            for k, v in train_losses.items():
                self.writer.add_scalar(f"epoch_train/{k}", v, epoch)
            self.writer.add_scalar("epoch_val/loss", val_loss, epoch)

            # 
            if self.scheduler is not None:
                self.scheduler.step()

            # 
            print(f"\nEpoch {epoch}/{num_epochs} - "
                  f"Train: {train_losses['total']:.4f} (js={train_losses['joystick']:.4f}, btn={train_losses['button']:.4f}) | "
                  f"Val: {val_loss:.4f} | Best: {self.best_val_loss:.4f} | Wait: {self.wait}/{self.early_stopping_patience}")

            # epoch
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # [CLEAN] Epoch
            if self.memory_clean_interval > 0:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # [FIRE] 
            if self.wait >= self.early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"[STOP] Early stopping triggered!")
                print(f"   No improvement for {self.early_stopping_patience} consecutive epochs")
                print(f"   Best validation loss: {self.best_val_loss:.4f} at epoch {epoch - self.early_stopping_patience}")
                print(f"{'='*60}")
                break

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total epochs trained: {epoch}")
        print(f"{'='*60}")
        self.writer.close()


def split_train_val_annotations(
    annotations_path: str,
    val_ratio: float = 0.1
) -> tuple:
    """"""

    # 
    if not os.path.exists(annotations_path):
        print(f"\n{'='*60}")
        print("[ERROR] Training data not found!")
        print(f"{'='*60}")
        print(f"\nMissing file: {annotations_path}")
        print("\nPlease generate training data first:")
        print("  python generate_synthetic_training_data.py")
        print("\nOr generate a small test dataset:")
        print("  python generate_synthetic_training_data.py --num_samples 1000")
        print("\nFor more information, see README.md")
        print(f"{'='*60}\n")
        raise FileNotFoundError(
            f"Training data not found: {annotations_path}\n"
            f"Please run: python generate_synthetic_training_data.py"
        )

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    video_ids = list(annotations.keys())
    num_val = int(len(video_ids) * val_ratio)

    # 
    import random
    random.shuffle(video_ids)

    val_videos = set(video_ids[:num_val])
    train_videos = set(video_ids[num_val:])

    train_annotations = {
        k: v for k, v in annotations.items() if k in train_videos
    }
    val_annotations = {
        k: v for k, v in annotations.items() if k in val_videos
    }

    # 
    train_ann_path = Path(annotations_path).parent / "train_annotations.json"
    val_ann_path = Path(annotations_path).parent / "val_annotations.json"

    with open(train_ann_path, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2)

    with open(val_ann_path, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2)

    print(f"  Train: {len(train_annotations)} | Val: {len(val_annotations)} videos")

    return str(train_ann_path), str(val_ann_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train SegFormer Action Parser")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu, default: auto)")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help=f"Number of data workers (default: {NUM_WORKERS})")
    # 
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training (AMP) to reduce memory usage")
    parser.add_argument("--memory_clean_interval", type=int, default=0,
                        help="Memory cleanup interval (batches, 0=disabled)")
    # 
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement, default: 10)")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum change to qualify as improvement (default: 0.001)")

    args = parser.parse_args()

    print("=" * 60)
    print("Training SegFormer Action Parser")
    print("  NitroGen Paper Implementation")
    print("=" * 60)

    # 
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 
    print(f"\nDevice: {device}")
    opts = []
    if args.mixed_precision and AMP_AVAILABLE:
        opts.append("")
    if args.memory_clean_interval > 0:
        opts.append(f"({args.memory_clean_interval})")
    if opts:
        print(f"  : {' | '.join(opts)}")

    # 
    if not os.path.exists("synthetic_data/train_annotations.json"):
        print("\n[1/4] Splitting train/val annotations...")
        train_ann_path, val_ann_path = split_train_val_annotations(
            "synthetic_data/annotations.json",
            val_ratio=0.1
        )
    else:
        train_ann_path = "synthetic_data/train_annotations.json"
        val_ann_path = "synthetic_data/val_annotations.json"
        print("\n[1/4] Using existing train/val split [OK]")

    # 
    print("[2/4] Creating data loaders...")
    train_loader, val_loader = create_train_val_dataloaders(
        data_dir=DATA_DIR,
        train_annotations=train_ann_path,
        val_annotations=val_ann_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=IMAGE_SIZE
    )

    print(f"  Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # 
    print("[3/4] Creating model...")
    model = NitroGenActionParser(
        num_buttons=16,
        joystick_grid_size=11
    ).to(device)
    print("  [OK] Model created")

    # 
    criterion = ActionParsingLoss(joystick_weight=1.0, button_weight=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    # 
    if len(train_loader) == 0:
        print("\n[ERROR] No training data available!")
        print("Please generate more training data:")
        print("  python generate_synthetic_training_data.py --num_samples 1000")
        raise ValueError("No training data available")

    #  ()
    num_steps = len(train_loader) * args.num_epochs // GRADIENT_ACCUMULATION_STEPS

    if num_steps == 0:
        print("\n[WARNING] Too few training steps for learning rate scheduling")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.num_epochs}")
        print("\nRecommendation: Generate at least 100 samples for testing")
        print("  python generate_synthetic_training_data.py --num_samples 100")
        num_steps = 1  # 

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0, 1 - step / num_steps)
    )

    # 
    print("[4/4] Initializing trainer...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=f"{OUTPUT_DIR}/{timestamp}",
        log_dir=f"{LOG_DIR}/{timestamp}",
        use_amp=args.mixed_precision,
        memory_clean_interval=args.memory_clean_interval,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta
    )
    print(f"  Output: {OUTPUT_DIR}/{timestamp}\n")

    # 
    trainer.train(train_loader, val_loader, args.num_epochs)


if __name__ == "__main__":
    main()
