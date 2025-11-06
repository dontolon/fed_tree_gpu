import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np
import os
import random
from PIL import Image

TREE_CLASS_SUBSETS = {
    "cifar100": ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    "inaturalist": []  # placeholder for generic folder datasets
}


def get_transform(model_type='mobilenet'):
    if model_type == 'mobilenet':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


class TreeSubsetDataset(Dataset):
    """
    Unified dataset class for:
      - CIFAR-100 (predefined or random subset)
      - Folder-based datasets (e.g., iNaturalist)

    ----------------------------------------------------------------------
    HOW TO ADAPT TO A CUSTOM FOLDER-BASED DATASET
    ----------------------------------------------------------------------
    Folder structure expected:
       root/
         ├── dataset_name/
         │     ├── train/ or train_mini/
         │     │     ├── <class_name_1>/img1.jpg ...
         │     │     ├── <class_name_2>/img2.jpg ...
         │     └── val/ (optional)
    
    To adapt for another dataset:
      1. Change the logic that identifies the data folder:
         → modify the search in ["train_mini", "train", "training"].
      2. Adjust the class filtering rule:
         → e.g., replace `'plantae' in d.lower()` with your own keyword filter,
            or remove it to include *all* folders.
      3. If class labels are encoded differently in folder names,
         edit the `extract_label()` function to parse your folder names.
      4. Optionally, change the top-N filtering logic if you prefer
         balanced sampling or random selection instead of largest classes.
    ----------------------------------------------------------------------
    """
    def __init__(self, dataset_name, root, train, transform,
                 download=True, num_classes=None, class_names=None,
                 random_subset=False):

        dataset_name = dataset_name.lower()
        self.transform = transform
        self.indices = []
        self.class_names = []
        self.label_map = {}

        # ------------------------------------------------------------------
        # Case 1: CIFAR-100
        # ------------------------------------------------------------------
        if dataset_name == "cifar100":
            base = datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
            all_classes = base.classes
            targets = base.targets

            # Select class subset
            if class_names is None:
                if num_classes is not None:
                    if random_subset:
                        class_names = random.sample(all_classes, num_classes)
                    else:
                        class_names = all_classes[:num_classes]
                else:
                    class_names = TREE_CLASS_SUBSETS["cifar100"]

            name_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
            self.original_class_indices = [name_to_idx[cls] for cls in class_names]
            self.label_map = {orig: new for new, orig in enumerate(self.original_class_indices)}

            self.indices = [i for i, label in enumerate(targets) if label in self.original_class_indices]
            self.base = base
            self.class_names = class_names
            self.getitem_func = self._getitem_cifar
            return  # done

        # ------------------------------------------------------------------
        # Case 2: Folder-based datasets (e.g., iNaturalist)
        # ------------------------------------------------------------------
        dataset_root = os.path.join(root, dataset_name)
        if not os.path.isdir(dataset_root):
            dataset_root = root  # fallback if placed directly in root

        # Locate subdirectory (train/validation)
        subdir = None
        for cand in ["train_mini", "train", "training"]:
            if os.path.isdir(os.path.join(dataset_root, cand)):
                subdir = cand
                break
        if not subdir:
            raise FileNotFoundError(f"No train folder found in {dataset_root}")

        data_path = os.path.join(dataset_root, subdir if train else "val")
        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"Expected folder not found: {data_path}")

        # --- Filter for Plantae folders only ---
        plantae_dirs = [d for d in os.listdir(data_path)
                        if os.path.isdir(os.path.join(data_path, d)) and "plantae" in d.lower()]

        if not plantae_dirs:
            raise RuntimeError(
                f"No folders containing 'Plantae' found in {data_path}. "
                "Expected structure like train_mini/<id>_Plantae_<species>/..."
            )

        # --- Count images per class ---
        class_counts = {}
        for d in plantae_dirs:
            class_dir = os.path.join(data_path, d)
            n_imgs = sum(
                fname.lower().endswith(('.jpg', '.jpeg', '.png'))
                for _, _, files in os.walk(class_dir)
                for fname in files
            )
            class_counts[d] = n_imgs

        # --- Select top-N classes by sample count ---
        if num_classes is not None and len(plantae_dirs) > num_classes:
            sorted_dirs = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
            top_dirs = [d for d, _ in sorted_dirs[:num_classes]]
            plantae_dirs = top_dirs

        # --- Build file list ---
        filtered_samples = []
        for d in plantae_dirs:
            class_dir = os.path.join(data_path, d)
            for root_dir, _, files in os.walk(class_dir):
                for fname in files:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filtered_samples.append((os.path.join(root_dir, fname), d))

        if not filtered_samples:
            raise RuntimeError(f"No image files found in Plantae folders under {data_path}.")

        # --- Derive class labels from folder names ---
        def extract_label(folder_name):
            # Customize this for your dataset’s naming convention
            parts = folder_name.lower().split("_")
            return parts[-2] if len(parts) >= 2 else folder_name.lower()

        label_map = {d: extract_label(d) for d in plantae_dirs}

        # --- Derive consistent label mapping ---
        if class_names is not None and len(class_names) > 0:
            unique_labels = class_names
        else:
            unique_labels = sorted(set(label_map.values()))

        class_to_idx = {cls: idx for idx, cls in enumerate(unique_labels)}

        self.samples = []
        for path, label in filtered_samples:
            lbl_name = label_map[label]
            if lbl_name not in class_to_idx:
                continue
            self.samples.append((path, class_to_idx[lbl_name]))

        self.class_names = unique_labels
        self.getitem_func = self._getitem_folder
        self.indices = list(range(len(self.samples)))

        # Print a small summary for clarity
        print(f"[TreeSubsetDataset] Loaded {len(self.class_names)} classes "
              f"({', '.join(self.class_names)}) with {len(self.samples)} images total.")

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def _getitem_cifar(self, idx):
        real_idx = self.indices[idx]
        img, orig_label = self.base[real_idx]
        new_label = self.label_map[orig_label]
        return img, new_label

    def _getitem_folder(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.getitem_func(idx)

    def get_class_indices(self):
        """Return dict: class_idx -> [dataset_indices]"""
        class_indices = defaultdict(list)
        for ds_idx, real_idx in enumerate(self.indices):
            if hasattr(self, "samples"):
                _, label = self.samples[real_idx]
            else:
                _, label = self.base[real_idx]
            class_indices[label].append(ds_idx)
        return class_indices
