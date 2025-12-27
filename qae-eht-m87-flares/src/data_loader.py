import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json

class M87FlareDataset(Dataset):
    """
    Custom PyTorch Dataset for M87* flare detection
    """
    
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        """
        Args:
            data_dir: Path to data/processed/ folder
            split: "train", "val", or "test" (BEWARE: train/val are NORMAL ONLY)
            transform: optional torch vision transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # load split metadata
        split_file = self.data_dir / "splits" / f"{split}_files.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file missing: {split_file}\n"
                f"Run notebooks/01_data_synthesis.ipynb with CORRECT split logic!"
            )
        
        with open(split_file, "r") as f:
            self.file_list = json.load(f)
        
        # train/val must be NORMAL ONLY
        if split in ["train", "val"]:
            flare_count = sum(1 for f in self.file_list if f["label"] == 1)
            if flare_count > 0:
                raise ValueError(
                    f"[ERROR] {split} set contains {flare_count} FLARE images!\n"
                    f"For anomaly detection, train/val must be PURE NORMAL data."
                )
        
        print(f"[M87FlareDataset] Loaded {len(self.file_list)} {split} samples")
        print(f"[M87FlareDataset] Flare count: {sum(f['label'] for f in self.file_list)}")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx: int):
        file_info = self.file_list[idx]
        file_path = self.data_dir / file_info["path"]
        
        try:
            image = np.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}")
        
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(file_info["label"], dtype=torch.long)
        return image, label


def create_dataloaders(
    data_dir: str = "data/processed", 
    batch_size: int = 64,
    num_workers: int = 4
):
    """
    Factory function with CORRECT split sizes for anomaly detection
    
    Train/Val are NORMAL ONLY—crucial for autoencoder anomaly detection paradigm.
    """
    
    train_loader = DataLoader(
        M87FlareDataset(data_dir, split="train"),
        batch_size=batch_size,
        shuffle=True,  # training needs randomness
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        M87FlareDataset(data_dir, split="val"),
        batch_size=batch_size,
        shuffle=False,  # consistent validation
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        M87FlareDataset(data_dir, split="test"),
        batch_size=batch_size,
        shuffle=False,  # deterministic evaluation
        num_workers=num_workers,
        pin_memory=False
    )
    
    # print split sizes for verification
    print("\n[SPLIT SUMMARY]")
    print(f"Train: {len(train_loader.dataset)} samples (normal only)")
    print(f"Val:   {len(val_loader.dataset)} samples (normal only)")
    print(f"Test:  {len(test_loader.dataset)} samples (normal + flare)")
    
    return train_loader, val_loader, test_loader


# self-test
if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders()
    
    images, labels = next(iter(train_loader))
    print(f"\n[SANITY CHECK] Batch shape: {images.shape}")
    print(f"[SANITY CHECK] Train labels: {labels} (should ALL be 0)")
    
    images, labels = next(iter(test_loader))
    print(f"[SANITY CHECK] Test labels sample: {labels[:20]} (should have 1s)")