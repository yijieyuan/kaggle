import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import KFold
import segmentation_models_pytorch_3d as smp
from tqdm import tqdm
import h5py

# ==================== Custom Loss Functions ====================

class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss.
    If targets are zero, the dice is zero.
    """
    def __init__(self, num_classes=13):
        super(BCEDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def dice_loss_per_class(self, pred, target, smooth=1e-5):
        """
        Calculate Dice loss for each class separately.
        If target is all zeros for a class, return 0.
        
        Returns:
            dice_losses: list of dice loss for each class
        """
        pred = torch.sigmoid(pred)
        dice_losses = []
        
        for c in range(self.num_classes):
            pred_c = pred[:, c, ...]
            target_c = target[:, c, ...]
            
            # Check if target is all zeros
            if target_c.sum() == 0:
                dice_losses.append(0.0)
            else:
                intersection = (pred_c * target_c).sum()
                dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
                dice_losses.append(1 - dice)
        
        return dice_losses
    
    def forward(self, pred, target, return_per_class=False):
        """
        Args:
            pred: (B, C, D, H, W) - logits
            target: (B, C, D, H, W) - one-hot encoded
            return_per_class: if True, return per-class losses
        """
        # BCE loss per class
        bce_loss = self.bce(pred, target)
        bce_per_class = bce_loss.mean(dim=(0, 2, 3, 4))  # Average over batch and spatial dims
        bce_total = bce_per_class.mean()
        
        # Dice loss per class - collect as tensors
        dice_per_class_batch = []
        for b in range(pred.shape[0]):
            dice_losses = self.dice_loss_per_class(pred[b:b+1], target[b:b+1])
            # Convert to tensor
            dice_losses_tensor = torch.tensor(dice_losses, dtype=torch.float32, device=pred.device)
            dice_per_class_batch.append(dice_losses_tensor)
        
        # Stack and average across batch
        dice_per_class_batch = torch.stack(dice_per_class_batch)  # (B, num_classes)
        dice_per_class = dice_per_class_batch.mean(dim=0)  # (num_classes,)
        
        dice_total = dice_per_class.mean()
        
        # Combined loss
        total_loss = bce_total + dice_total
        
        if return_per_class:
            return total_loss, bce_per_class, dice_per_class
        return total_loss


# ==================== Dataset ====================

class HDF5Dataset(Dataset):
    """
    Dataset for loading HDF5 files with volumes and masks.
    """
    def __init__(self, vol_files, mask_files, num_classes=13):
        self.vol_files = vol_files
        self.mask_files = mask_files
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.vol_files)
    
    def __getitem__(self, idx):
        # Load volume
        with h5py.File(self.vol_files[idx], 'r') as f:
            volume = f['raw'][:]
        
        # Load mask
        with h5py.File(self.mask_files[idx], 'r') as f:
            mask = f['label'][:]
        
        # Convert to torch tensors
        volume = torch.from_numpy(volume).float()
        mask = torch.from_numpy(mask).long()
        
        # Add channel dimension to volume if needed
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # (1, D, H, W)
        
        # Convert mask to one-hot encoding
        mask_one_hot = torch.zeros(self.num_classes, *mask.shape)
        for c in range(self.num_classes):
            mask_one_hot[c] = (mask == c).float()
        
        return volume, mask_one_hot


# ==================== Training Functions ====================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch and print losses per batch and summarize per epoch.
    """
    model.train()
    
    epoch_bce_per_class = []
    epoch_dice_per_class = []
    epoch_total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    
    for batch_idx, (volumes, masks) in enumerate(pbar):
        volumes = volumes.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(volumes)
        
        # Calculate loss with per-class details
        loss, bce_per_class, dice_per_class = criterion(outputs, masks, return_per_class=True)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store losses
        epoch_total_loss += loss.item()
        epoch_bce_per_class.append(bce_per_class.detach().cpu().numpy())
        epoch_dice_per_class.append(dice_per_class.detach().cpu().numpy())
        
        # Print batch losses
        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Total Loss: {loss.item():.4f}")
        print("BCE Loss per class:")
        for c in range(len(bce_per_class)):
            print(f"  Class {c}: {bce_per_class[c].item():.4f}")
        print("Dice Loss per class:")
        for c in range(len(dice_per_class)):
            print(f"  Class {c}: {dice_per_class[c]:.4f}")
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate epoch averages
    avg_total_loss = epoch_total_loss / len(dataloader)
    avg_bce_per_class = np.mean(epoch_bce_per_class, axis=0)
    avg_dice_per_class = np.mean(epoch_dice_per_class, axis=0)
    
    # Print epoch summary
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch} SUMMARY")
    print(f"{'='*60}")
    print(f"Average Total Loss: {avg_total_loss:.4f}")
    print("\nAverage BCE Loss per class:")
    for c in range(len(avg_bce_per_class)):
        print(f"  Class {c}: {avg_bce_per_class[c]:.4f}")
    print("\nAverage Dice Loss per class:")
    for c in range(len(avg_dice_per_class)):
        print(f"  Class {c}: {avg_dice_per_class[c]:.4f}")
    print(f"{'='*60}\n")
    
    return avg_total_loss, avg_bce_per_class, avg_dice_per_class


def validate_epoch(model, dataloader, criterion, device, epoch):
    """
    Validate for one epoch.
    """
    model.eval()
    
    epoch_bce_per_class = []
    epoch_dice_per_class = []
    epoch_total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Validation')
    
    with torch.no_grad():
        for batch_idx, (volumes, masks) in enumerate(pbar):
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(volumes)
            
            # Calculate loss with per-class details
            loss, bce_per_class, dice_per_class = criterion(outputs, masks, return_per_class=True)
            
            # Store losses
            epoch_total_loss += loss.item()
            epoch_bce_per_class.append(bce_per_class.detach().cpu().numpy())
            epoch_dice_per_class.append(dice_per_class.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate epoch averages
    avg_total_loss = epoch_total_loss / len(dataloader)
    avg_bce_per_class = np.mean(epoch_bce_per_class, axis=0)
    avg_dice_per_class = np.mean(epoch_dice_per_class, axis=0)
    
    # Print validation summary
    print(f"\n{'='*60}")
    print(f"VALIDATION EPOCH {epoch} SUMMARY")
    print(f"{'='*60}")
    print(f"Average Total Loss: {avg_total_loss:.4f}")
    print("\nAverage BCE Loss per class:")
    for c in range(len(avg_bce_per_class)):
        print(f"  Class {c}: {avg_bce_per_class[c]:.4f}")
    print("\nAverage Dice Loss per class:")
    for c in range(len(avg_dice_per_class)):
        print(f"  Class {c}: {avg_dice_per_class[c]:.4f}")
    print(f"{'='*60}\n")
    
    return avg_total_loss, avg_bce_per_class, avg_dice_per_class


# ==================== Main Training Pipeline ====================

def train_fold(fold, train_vol_files, train_mask_files, val_vol_files, val_mask_files, 
               num_classes, num_epochs, device, save_dir):
    """
    Train a single fold.
    """
    print(f"\n{'#'*60}")
    print(f"# TRAINING FOLD {fold}")
    print(f"{'#'*60}\n")
    
    # Create datasets
    train_dataset = HDF5Dataset(train_vol_files, train_mask_files, num_classes)
    val_dataset = HDF5Dataset(val_vol_files, val_mask_files, num_classes)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights=None,
        decoder_channels=(256, 128, 64, 32, 16),
        in_channels=1,
        classes=num_classes,
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = BCEDiceLoss(num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_bce, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_bce, val_dice = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'best_model_fold{fold}.pth'))
            print(f"✓ Saved best model for fold {fold} (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_fold{fold}_epoch{epoch}.pth'))
    
    return best_val_loss


def main():
    """
    Main training pipeline with 3-fold cross-validation.
    """
    # Configuration
    VOL_DIR = "./hdf5_vol"
    MASK_DIR = "./hdf5_mask"
    NUM_CLASSES = 13
    NUM_FOLDS = 3
    NUM_EPOCHS = 50
    SAVE_DIR = "./checkpoints"
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all files
    vol_files = sorted(list(Path(VOL_DIR).glob("*.h5")))
    mask_files = sorted(list(Path(MASK_DIR).glob("*.h5")))
    
    print(f"Found {len(vol_files)} volume files")
    print(f"Found {len(mask_files)} mask files")
    
    # Verify pairing
    assert len(vol_files) == len(mask_files), "Volume and mask counts don't match!"
    
    # Convert to string paths
    vol_files = [str(f) for f in vol_files]
    mask_files = [str(f) for f in mask_files]
    
    # K-Fold split
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(vol_files), 1):
        # Split files
        train_vol_files = [vol_files[i] for i in train_idx]
        train_mask_files = [mask_files[i] for i in train_idx]
        val_vol_files = [vol_files[i] for i in val_idx]
        val_mask_files = [mask_files[i] for i in val_idx]
        
        # Train fold
        best_val_loss = train_fold(
            fold=fold,
            train_vol_files=train_vol_files,
            train_mask_files=train_mask_files,
            val_vol_files=val_vol_files,
            val_mask_files=val_mask_files,
            num_classes=NUM_CLASSES,
            num_epochs=NUM_EPOCHS,
            device=device,
            save_dir=SAVE_DIR
        )
        
        fold_results.append(best_val_loss)
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS - 3-FOLD CROSS-VALIDATION")
    print("="*60)
    for fold, val_loss in enumerate(fold_results, 1):
        print(f"Fold {fold} - Best Validation Loss: {val_loss:.4f}")
    print(f"\nAverage Validation Loss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()