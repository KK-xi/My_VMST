import numpy as np
from os import listdir
from os.path import join

class My_data:
    def __init__(self, root, flags):
        self.classes = listdir(root)
        self.flags = flags

        self.files = []
        self.labels = []

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """

        label = self.labels[idx]
        f = self.files[idx]
        data = np.load(f)
        voxel_coords = data["coords"].astype(np.float32)
        voxel_features = data["feature"].astype(np.float32)

        return voxel_coords, voxel_features, label