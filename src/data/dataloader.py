"""
Module providing PyTorch Dataset and DataLoader integration.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.dataset import PathBasedDataset

class FacialExpressionDataset(Dataset):
    """
    PyTorch Dataset wrapper for the PathBasedDataset.
    Converts numpy arrays to PyTorch tensors and reshapes to [C, H, W].
    """
    def __init__(self, paths_by_class: dict, augment: bool = False):
        self.base_dataset = PathBasedDataset(paths_by_class, augment=augment)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # image_array is (48, 48, 1) numpy array
        image_array, label = self.base_dataset[idx]
        
        # Convert to tensor
        img_tensor = torch.from_numpy(image_array)
        
        # Reshape [H, W, C] -> [C, H, W]
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

def get_dataloaders(train_split, val_split, test_split, batch_size=64, augment_train=False):
    """
    Creates DataLoaders for train, validation, and test splits.
    """
    train_dataset = FacialExpressionDataset(train_split, augment=augment_train)
    val_dataset = FacialExpressionDataset(val_split, augment=False)
    test_dataset = FacialExpressionDataset(test_split, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
