"""
train_minimal.py - Minimal training script with checkpoints and logging
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from dataset import create_dataloaders
from model import Model

def dice_loss(pred, target, smooth=1e-6):
    """Compute DICE loss for segmentation"""
    # if target is all zeros, return 0
    if target.sum() == 0:
        return 0.0
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def log_and_print(message, log_file):
    """Print message and write to log file"""
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

def train_epoch(model, train_loader, optimizer, device, log_file):
    """Train for one epoch"""
    model.train()
    classification_loss = nn.BCEWithLogitsLoss(reduction='none')
    segmentation_bce = nn.BCEWithLogitsLoss()
   
    total_loss = 0
    total_seg_loss = 0
    total_class_losses = torch.zeros(14, device=device)
   
    for batch_idx, (slabs, attention_maps, labels) in enumerate(train_loader):
        slabs = slabs.to(device)
        attention_maps = attention_maps.to(device)
        labels = labels.to(device)
       
        optimizer.zero_grad()
        cls_output, seg_output = model(slabs)
       
        # Classification loss
        class_losses = classification_loss(cls_output, labels).mean(dim=0)
        loss_cls = class_losses.mean()
        
        # Segmentation loss (BCE + DICE) with 0.25 weight
        loss_seg = 0
        if seg_output is not None:
            seg_bce_loss = segmentation_bce(seg_output, attention_maps)
            seg_dice_loss = dice_loss(seg_output, attention_maps)
            loss_seg = 0.25 * (seg_bce_loss + seg_dice_loss)
        
        loss = loss_cls + loss_seg
       
        loss.backward()
        optimizer.step()
       
        total_loss += loss.item()
        total_seg_loss += loss_seg.item() if seg_output is not None else 0
        total_class_losses += class_losses
        
        # Print batch info
        bce_str = " | ".join([f"{loss:.5f}" for loss in class_losses])
        message = f"Train B{batch_idx:03d} BCE: {bce_str} | Mean: {loss_cls:.5f} | Seg: {loss_seg:.5f} BCE: {seg_bce_loss:.5f} DICE: {seg_dice_loss:.5f} | Total: {loss:.5f}"
        print(message, end='\r')
   
    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = total_seg_loss / len(train_loader)
    avg_class_losses = total_class_losses / len(train_loader)
   
    return avg_loss, avg_class_losses, avg_seg_loss

def validate_volume(model, val_loader, device, log_file):
    """Validate on volume-level data with max aggregation"""
    model.eval()
    model.mask_head = False  # Disable segmentation head during volume validation
    
    all_volume_preds, all_volume_labels = [], []
    
    with torch.no_grad():
        for vol_idx, volume_data in enumerate(val_loader):
            slabs = volume_data['slabs'].squeeze(0)  # Remove batch dim: [num_slabs, 3, 384, 384]
            label = volume_data['label'].squeeze(0)  # Remove batch dim: [14]
            
            # Process slabs in batches of 16
            slab_predictions = []
            for i in range(0, len(slabs), 16):
                batch_slabs = slabs[i:i+16].to(device)
                cls_output, _ = model(batch_slabs)
                slab_predictions.append(torch.sigmoid(cls_output))
            
            # Concatenate all slab predictions and apply max aggregation
            all_slab_preds = torch.cat(slab_predictions, dim=0)  # [num_slabs, 14]
            volume_pred = torch.max(all_slab_preds, dim=0)[0]     # [14]

            # Collect volume-level predictions and labels
            all_volume_preds.append(volume_pred.cpu())
            all_volume_labels.append(label.cpu())
            
            # Compute cumulative AUC
            cumulative_preds = torch.stack(all_volume_preds).numpy()
            cumulative_labels = torch.stack(all_volume_labels).numpy()
            
            cumulative_aucs = []
            for i in range(14):
                if len(np.unique(cumulative_labels[:, i])) > 1:
                    auc = roc_auc_score(cumulative_labels[:, i], cumulative_preds[:, i])
                    cumulative_aucs.append(auc)
                else:
                    cumulative_aucs.append(0.0)
            
            # Print volume info
            auc_str = " | ".join([f"{auc:.3f}" for auc in cumulative_aucs])
            mean_auc = np.mean(cumulative_aucs)
            # Weighted AUC: weight 1 for classes 0-12, weight 13 for class 13
            weights = [1] * 13 + [13]
            weighted_auc = sum(auc * w for auc, w in zip(cumulative_aucs, weights)) / sum(weights)
            message = f"Vol1  V{vol_idx:03d} AUC: {auc_str} | Mean: {mean_auc:.3f} | Weighted: {weighted_auc:.3f}"
            print(message, end='\r')
    
    model.mask_head = True  # Re-enable segmentation head after validation
    return cumulative_aucs

def validate(model, val_loader, device, log_file):
    """Validate"""
    model.eval()
    model.mask_head = False  # Disable segmentation head during validation
    classification_loss = nn.BCEWithLogitsLoss(reduction='none')
   
    total_loss = 0
    total_class_losses = torch.zeros(14, device=device)
    all_outputs, all_labels = [], []
   
    with torch.no_grad():
        for batch_idx, (slabs, labels) in enumerate(val_loader):
            slabs = slabs.to(device)
            labels = labels.to(device)
           
            cls_output, _ = model(slabs)

            # Calculate per-class losses
            class_losses = classification_loss(cls_output, labels).mean(dim=0)
            loss = class_losses.mean()
           
            total_loss += loss.item()
            total_class_losses += class_losses
            
            # Collect for AUC (accumulate all predictions)
            all_outputs.append(torch.sigmoid(cls_output).cpu())
            all_labels.append(labels.cpu())
            
            # Compute cumulative AUC using all predictions so far
            cumulative_outputs = torch.cat(all_outputs, dim=0).numpy()
            cumulative_labels = torch.cat(all_labels, dim=0).numpy()
            
            cumulative_aucs = []
            for i in range(14):
                if len(np.unique(cumulative_labels[:, i])) > 1:
                    auc = roc_auc_score(cumulative_labels[:, i], cumulative_outputs[:, i])
                    cumulative_aucs.append(auc)
                else:
                    cumulative_aucs.append(0.0)
            
            # Print batch info with cumulative AUC
            bce_str = " | ".join([f"{loss:.5f}" for loss in class_losses])
            auc_str = " | ".join([f"{auc:.3f}" for auc in cumulative_aucs])
            mean_auc = np.mean(cumulative_aucs)
            # Weighted AUC
            weights = [1] * 13 + [13]
            weighted_auc = sum(auc * w for auc, w in zip(cumulative_aucs, weights)) / sum(weights)
            message = f"Val2  B{batch_idx:03d} BCE: {bce_str} | Mean: {loss:.5f} | AUC: {auc_str} | Mean: {mean_auc:.3f} | W: {weighted_auc:.3f}"
            print(message, end='\r')
   
    avg_loss = total_loss / len(val_loader)
    avg_class_losses = total_class_losses / len(val_loader)
    
    # Final epoch AUC (same as the last cumulative computation)
    aucs = cumulative_aucs
    
    model.mask_head = True  # Re-enable segmentation head after validation
    
    return avg_loss, avg_class_losses, aucs

def print_column_headers(log_file):
    """Print column headers for better readability"""
    class_names = [f"C{i:02d}" for i in range(14)]  # C00, C01, ..., C13
    header = "      Phase    " + " | ".join([f"{name:>7}" for name in class_names]) + " | Mean   | Weighted"
    separator = "="*len(header)
    
    log_and_print("", log_file)
    log_and_print(separator, log_file)
    log_and_print(header, log_file)
    log_and_print(separator, log_file)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    return filepath

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    os.makedirs('./output/checkpoint', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    
    # Initialize logging
    log_file = open('./output/log.txt', 'w')
    log_and_print(f"Training started on device: {device}", log_file)
    log_and_print(f"Torch version: {torch.__version__}", log_file)
   
    # Create data loaders
    log_and_print("Creating data loaders...", log_file)
    train_loader, val_loader1, val_loader2 = create_dataloaders(
        fold=0,
        train_batch_size=4,
        val_batch_size=16,
        num_workers=0
    )
    log_and_print("Data loaders created.", log_file)
   
    # Initialize model with segmentation head enabled
    log_and_print("Initializing model with segmentation head...", log_file)
    model = Model(
        pre="coat_lite_medium_384x384_f9129688.pth",
        # pre=None,
        num_classes=14,
        ps=0.1,
        mask_head=True  # Enable segmentation head
    ).to(device)
    log_and_print("Model initialized with segmentation head.", log_file)
   
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    
    # Track best validation loss
    best_val_loss = float('inf')
    best_weighted_auc = 0
   
    # Training loop
    for epoch in range(1000):
        log_and_print(f"\nEpoch {epoch+1}/1000", log_file)
        print_column_headers(log_file)
       
        # Train with segmentation
        train_loss, train_class_losses, train_seg_loss = train_epoch(model, train_loader, optimizer, device, log_file)
        
        # Volume-level validation
        val1_aucs = validate_volume(model, val_loader1, device, log_file)
        
        # Slab-level validation
        val_loss, val_class_losses, val_aucs = validate(model, val_loader2, device, log_file)
        
        # Calculate weighted AUC
        weights = [1] * 13 + [13]
        val1_weighted_auc = sum(auc * w for auc, w in zip(val1_aucs, weights)) / sum(weights)
        val_weighted_auc = sum(auc * w for auc, w in zip(val_aucs, weights)) / sum(weights)
        
        # Print epoch summary with proper formatting
        train_bce_str = " | ".join([f"{loss:.5f}" for loss in train_class_losses])
        val_bce_str = " | ".join([f"{loss:.5f}" for loss in val_class_losses])
        val_auc_str = " | ".join([f"{auc:.3f}" for auc in val_aucs])
        val1_auc_str = " | ".join([f"{auc:.3f}" for auc in val1_aucs])
        
        mean_train_loss = train_loss
        mean_val_loss = val_loss
        mean_val_auc = np.mean(val_aucs)
        mean_val1_auc = np.mean(val1_aucs)
        
        log_and_print(f"\nTrain BCE: {train_bce_str} | {mean_train_loss:.5f} | Seg: {train_seg_loss:.5f}", log_file)
        log_and_print(f"Vol1  AUC: {val1_auc_str} | {mean_val1_auc:.3f} | {val1_weighted_auc:.3f}", log_file)
        log_and_print(f"Val2  BCE: {val_bce_str} | {mean_val_loss:.5f}", log_file)
        log_and_print(f"Val2  AUC: {val_auc_str} | {mean_val_auc:.3f} | {val_weighted_auc:.3f}", log_file)
        log_and_print("-" * 140, log_file)
        
        # Save checkpoint every epoch
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch + 1, val_loss, 
            './output/checkpoint', f'epoch_{epoch + 1:04d}.pt'
        )
        log_and_print(f"Saved checkpoint: {checkpoint_path}", log_file)
        
        # Track and save best model based on weighted AUC
        if val_weighted_auc > best_weighted_auc:
            best_weighted_auc = val_weighted_auc
            best_val_loss = val_loss
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch + 1, val_loss, 
                './output/checkpoint', 'best.pt'
            )
            log_and_print(f"New best weighted AUC: {val_weighted_auc:.5f} - Saved best checkpoint", log_file)
    
    log_and_print(f"\nTraining completed. Best weighted AUC: {best_weighted_auc:.5f}, Best val loss: {best_val_loss:.5f}", log_file)
    log_file.close()

if __name__ == "__main__":
    main()