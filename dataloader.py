import torch

class VAMPNetDataset(torch.utils.data.Dataset):
    def __init__(self, data, lag_time=1):
        """
        Create a dataset of time-lagged pairs.

        Args:
            data: Sequential data
            lag_time: Time lag between pairs
        """
        self.x_t0 = data[:len(data) - lag_time]
        self.x_t1 = data[lag_time:]

    def __len__(self):
        return len(self.x_t0)

    def __getitem__(self, idx):
        return self.x_t0[idx], self.x_t1[idx]


# Usage example:
def create_vampnet_dataloader(data, lag_time=1, batch_size=32, shuffle=True):
    dataset = VAMPNetDataset(data, lag_time)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
