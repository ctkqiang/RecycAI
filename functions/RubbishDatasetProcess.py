try:
    import os
    import json
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision.transforms import functional as F
except ImportError as e:
    raise e


class RubbishDatasetProcess(Dataset):
    def __init__(self, img_dir, annotations):
        self.img_dir = img_dir
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.img_dir, self.annotations[idx]['image'])
        img = Image.open(img_path).convert("RGB")

        # Convert image to tensor
        img = F.to_tensor(img)

        # Get bounding boxes and labels
        boxes = self.annotations[idx]['boxes']
        labels = self.annotations[idx]['labels']

        # Prepare the target dictionary
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return img, target
