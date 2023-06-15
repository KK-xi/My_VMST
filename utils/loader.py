import torch
import numpy as np
<<<<<<< HEAD
=======

>>>>>>> f16047b (first commit)
from torch.utils.data.dataloader import default_collate

class Loader:
    def __init__(self, dataset, flags, device, shuffle=True):
        self.device = device
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=shuffle,
                                             num_workers=flags.num_workers, pin_memory=flags.pin_memory,
                                             collate_fn=self.collate_events)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)

    def collate_events(self, data):
        labels = []
        coords, points = [], []

        for i, d in enumerate(data):
            labels.append(d[2])
            pos = d[0]
            coords.append(pos)
            points.append(d[1])

        labels = default_collate(labels)

        coords = torch.from_numpy(np.stack(coords, 0))
        points = torch.from_numpy(np.stack(points, 0))

        return coords, points, labels