import os

from torch.utils.data import Dataset

from app.preprocess import preprocess


class CustomDataset(Dataset):
    def __init__(self, data_folder_path):
        self.image_paths = []
        self.root = data_folder_path
        for path in os.listdir(data_folder_path):
            if not os.path.isdir(os.path.join(self.root, path)):
                self.image_paths.append(os.path.join(self.root, path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return preprocess(self.image_paths[idx])
