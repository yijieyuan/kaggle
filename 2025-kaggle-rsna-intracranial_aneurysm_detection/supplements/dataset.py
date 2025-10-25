"""
dataset.py - Dataset classes using pregenerated slabs
Training: Randomly sample series and slab with specific labels
Validation: Process all slabs without attention maps
ValidationDataset2: Randomly sample one slab per series for validation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
from tqdm import tqdm

# ImageNet normalization for pre-trained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TrainDataset(Dataset):
    """Training dataset using pregenerated slabs - Randomly samples one slab per __getitem__ call"""
    
    def __init__(self, slab_dir, att_map_dir, label_dir, cv_splits_dir, fold, normalize=True, positive_weight=0.5):
        self.slab_dir = Path(slab_dir)
        self.att_map_dir = Path(att_map_dir)
        self.label_dir = Path(label_dir)
        self.normalize = normalize
        self.positive_weight = positive_weight
        
        # Load split file
        split_file = Path(cv_splits_dir) / f"fold_{fold}.json"
        with open(split_file, 'r') as f:
            splits = json.load(f)
        train_series = splits['train']
        
        # Invalid series to exclude
        self.invalid_series = [
            "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
            "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068"
        ]
        
        # Build list of valid series and their slabs
        self.positive_series = []
        self.negative_series = []
        self.series_slabs = {}
        
        print("Loading training data...")
        for series_uid in tqdm(train_series):
            if series_uid in self.invalid_series:
                continue
                
            series_slab_dir = self.slab_dir / series_uid
            if not series_slab_dir.exists():
                continue
            
            # Check if positive or negative
            volume_label_file = series_slab_dir / "label.npy"
            if volume_label_file.exists():
                volume_label = np.load(volume_label_file)
                is_positive = volume_label[-1] > 0
            else:
                series_label_dir = self.label_dir / series_uid
                is_positive = series_label_dir.exists()
            
            # Get all slab files
            slab_files = sorted([f for f in series_slab_dir.glob("[0-9]*.npy")])
            
            if slab_files:
                self.series_slabs[series_uid] = {
                    'slab_files': slab_files,
                    'is_positive': is_positive
                }
                
                if is_positive:
                    self.positive_series.append(series_uid)
                else:
                    self.negative_series.append(series_uid)
        
        self.all_series = self.positive_series + self.negative_series
        print(f"Training dataset: {len(self.positive_series)} positive, {len(self.negative_series)} negative series")
    
    def _normalize_slab(self, slab):
        if not self.normalize:
            return slab
        for c in range(3):
            slab[c] = (slab[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return slab
    
    def __len__(self):
        return (len(self.negative_series) + len(self.positive_series)) * 2
    
    def __getitem__(self, idx):
        series_uid = self.all_series[idx % len(self.all_series)]
        series_data = self.series_slabs[series_uid]
        
        # Randomly select a slab
        slab_file = random.choice(series_data['slab_files'])
        start_z = int(slab_file.stem)
        
        # Load slab
        slab = np.load(slab_file)
        slab = slab.astype(np.float32) / 255.0
        slab = self._normalize_slab(slab)
        
        # Load or create label
        if series_data['is_positive']:
            label_file = self.label_dir / series_uid / f"{start_z}.npy"
            assert label_file.exists(), f"Label file not found: {label_file}"
            label = np.load(label_file).astype(np.float32)
        else:
            label = np.zeros(14, dtype=np.float32)
        
        # Load or create attention map
        if series_data['is_positive']:
            att_file = self.att_map_dir / series_uid / f"{start_z}.npy"
            assert att_file.exists(), f"Attention file not found: {att_file}"
            attention = np.load(att_file).astype(np.float32)
        else:
            attention = np.zeros((3, 384, 384), dtype=np.float32)
        
        return (torch.from_numpy(slab).float(),
                torch.from_numpy(attention).float(),
                torch.from_numpy(label).float())


class ValidationDataset(Dataset):
    """Validation dataset using pregenerated slabs - Returns entire volume without attention maps"""
    
    def __init__(self, val_dir, cv_splits_dir, fold, normalize=True):
        self.val_dir = Path(val_dir)
        self.normalize = normalize
        
        # Load split file
        split_file = Path(cv_splits_dir) / f"fold_{fold}.json"
        with open(split_file, 'r') as f:
            splits = json.load(f)
        val_series = splits['val']
        
        # Invalid series to exclude
        self.invalid_series = [
            "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
            "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068"
        ]
        
        # Build list of valid series
        self.valid_series = []
        for series_uid in val_series:
            if series_uid in self.invalid_series:
                continue
                
            series_dir = self.val_dir / series_uid
            slabs_file = series_dir / "slabs.npy"
            label_file = series_dir / "label.npy"
            
            if slabs_file.exists() and label_file.exists():
                self.valid_series.append(series_uid)
        
        self.valid_series = self.valid_series[:200]
        print(f"Validation dataset: {len(self.valid_series)} volumes")
    
    def _normalize_slab(self, slab):
        if not self.normalize:
            return slab
        for c in range(3):
            slab[c] = (slab[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return slab
    
    def __len__(self):
        return len(self.valid_series)
    
    def __getitem__(self, idx):
        series_uid = self.valid_series[idx]
        series_dir = self.val_dir / series_uid
        
        # Load slabs and label
        slabs = np.load(series_dir / "slabs.npy")
        label = np.load(series_dir / "label.npy")
        
        # Normalize slabs
        normalized_slabs = []
        for slab in slabs:
            slab_float = slab.astype(np.float32) / 255.0
            slab_normalized = self._normalize_slab(slab_float)
            normalized_slabs.append(slab_normalized)
        
        slabs_tensor = torch.stack([torch.from_numpy(s).float() for s in normalized_slabs])
        label_tensor = torch.from_numpy(label.astype(np.float32)).float()
        
        return {
            'slabs': slabs_tensor,
            'label': label_tensor,
            'series_uid': series_uid,
            'num_slabs': len(slabs)
        }


class ValidationDataset2(Dataset):
    """Validation dataset using pregenerated slabs - Randomly selects one slab per series with manual seed"""
    
    def __init__(self, slab_dir, label_dir, cv_splits_dir, fold, normalize=True, seed=42):
        self.slab_dir = Path(slab_dir)
        self.label_dir = Path(label_dir)
        self.normalize = normalize
        
        # Set manual seed for reproducible slab selection
        random.seed(seed)
        np.random.seed(seed)
        
        # Load split file
        split_file = Path(cv_splits_dir) / f"fold_{fold}.json"
        with open(split_file, 'r') as f:
            splits = json.load(f)
        val_series = splits['val']
        
        # Invalid series to exclude
        self.invalid_series = [
            "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
            "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068"
        ]
        
        # Build list of valid series and randomly select one slab per series
        self.series_data = []
        
        print("Loading validation slab data...")
        for series_uid in tqdm(val_series):
            if series_uid in self.invalid_series:
                continue
                
            series_slab_dir = self.slab_dir / series_uid
            if not series_slab_dir.exists():
                continue
            
            # Check if positive or negative
            volume_label_file = series_slab_dir / "label.npy"
            if volume_label_file.exists():
                volume_label = np.load(volume_label_file)
                is_positive = volume_label[-1] > 0
            else:
                series_label_dir = self.label_dir / series_uid
                is_positive = series_label_dir.exists()
            
            # Get all slab files
            slab_files = sorted([f for f in series_slab_dir.glob("[0-9]*.npy")])
            
            if slab_files:
                # Randomly select one slab file with manual seed
                selected_slab = random.choice(slab_files)
                
                self.series_data.append({
                    'series_uid': series_uid,
                    'slab_file': selected_slab,
                    'is_positive': is_positive
                })
        
        print(f"Validation slab dataset: {len(self.series_data)} slabs")
    
    def _normalize_slab(self, slab):
        if not self.normalize:
            return slab
        for c in range(3):
            slab[c] = (slab[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return slab
    
    def __len__(self):
        return len(self.series_data)
    
    def __getitem__(self, idx):
        data = self.series_data[idx]
        series_uid = data['series_uid']
        slab_file = data['slab_file']
        is_positive = data['is_positive']
        
        start_z = int(slab_file.stem)
        
        # Load slab
        slab = np.load(slab_file)
        slab = slab.astype(np.float32) / 255.0
        slab = self._normalize_slab(slab)
        
        # Load or create label
        if is_positive:
            label_file = self.label_dir / series_uid / f"{start_z}.npy"
            assert label_file.exists(), f"Label file not found: {label_file}"
            label = np.load(label_file).astype(np.float32)
        else:
            label = np.zeros(14, dtype=np.float32)
        
        return (torch.from_numpy(slab).float(),
                torch.from_numpy(label).float())


def create_dataloaders(slab_dir='E:/kaggle-rsna/pre_train/slab',
                       att_map_dir='E:/kaggle-rsna/pre_train/att_map',
                       label_dir='E:/kaggle-rsna/pre_train/label',
                       val_dir='E:/kaggle-rsna/pre_valid',
                       cv_splits_dir='../../cv_splits',
                       fold=0,
                       train_batch_size=4,
                       val_batch_size=4,
                       num_workers=0,
                       normalize=True,
                       positive_weight=0.5,
                       val_seed=42):
    """Create train, validation, and validation2 dataloaders from pregenerated data"""
    
    # Training dataset
    train_dataset = TrainDataset(
        slab_dir=slab_dir,
        att_map_dir=att_map_dir,
        label_dir=label_dir,
        cv_splits_dir=cv_splits_dir,
        fold=fold,
        normalize=normalize,
        positive_weight=positive_weight
    )
    
    # Validation dataset (full volumes)
    val_dataset = ValidationDataset(
        val_dir=val_dir,
        cv_splits_dir=cv_splits_dir,
        fold=fold,
        normalize=normalize
    )
    
    # Validation dataset 2 (single slabs)
    val2_dataset = ValidationDataset2(
        slab_dir=slab_dir,
        label_dir=label_dir,
        cv_splits_dir=cv_splits_dir,
        fold=fold,
        normalize=normalize,
        seed=val_seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # One volume at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val2_loader = DataLoader(
        val2_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"\nDataLoaders created for fold {fold}:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} volumes")
    print(f"  Val2: {len(val2_dataset)} slabs, {len(val2_loader)} batches")
    
    return train_loader, val_loader, val2_loader


if __name__ == "__main__":
    # Test the datasets
    print("Testing pregenerated datasets...")
    
    train_loader, val_loader, val2_loader = create_dataloaders(
        fold=0,
        train_batch_size=4,
        val_batch_size=4,
        num_workers=0,
        positive_weight=0.5
    )
    
    # Test training batch
    print("\nTesting training loader...")
    for i, (slabs, attentions, labels) in enumerate(train_loader):
        print(f"Batch {i}: Slabs {slabs.shape}, Attention {attentions.shape}, Labels {labels.shape}")
        print(f"  Positive samples: {(labels[:, -1] > 0).sum().item()}/{labels.shape[0]}")
        break
    
    # Test validation batch
    print("\nTesting validation loader...")
    for i, volume_data in enumerate(val_loader):
        print(f"Volume {i}: Series {volume_data['series_uid'][0][:20]}...")
        print(f"  Slabs {volume_data['slabs'].shape}, Label {volume_data['label'].shape}")
        break
    
    # Test validation2 batch
    print("\nTesting validation2 loader...")
    for i, (slabs, labels) in enumerate(val2_loader):
        print(f"Batch {i}: Slabs {slabs.shape}, Labels {labels.shape}")
        print(f"  Positive samples: {(labels[:, -1] > 0).sum().item()}/{labels.shape[0]}")
        break
    
    print("\n✓ Dataset testing complete!")