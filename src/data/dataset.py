"""
Module providing a basic path-based dataset structure to avoid loading all images into RAM.
"""
from typing import List, Tuple
from src.data.config import REVERSE_CLASS_MAPPING
from src.preprocessing.transforms import load_and_preprocess_image

class PathBasedDataset:
    """
    A lightweight dataset class that stores paths and labels, 
    and loads/preprocesses images lazily when requested.
    """
    def __init__(self, paths_by_class: dict, augment: bool = False):
        self.augment = augment
        self.samples: List[Tuple[str, int]] = []
        for class_name, paths in paths_by_class.items():
            class_idx = REVERSE_CLASS_MAPPING[class_name]
            for path in paths:
                self.samples.append((path, class_idx))
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[object, int]:
        """
        Returns the preprocessed image array and its integer label.
        """
        path, label = self.samples[idx]
        image_array = load_and_preprocess_image(path, augment=self.augment)
        return image_array, label
