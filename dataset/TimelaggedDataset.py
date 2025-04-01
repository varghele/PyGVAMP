import torch

class TimelaggedDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, lag_time):
        """
        Create a dataset of time-lagged pairs from sequential data.

        Args:
            data_list: List of sequential data points
            lag_time: Time lag between pairs
        """
        self.n_samples = len(data_list) - lag_time
        self.data_t0 = data_list[:self.n_samples]
        self.data_t1 = data_list[lag_time:lag_time + self.n_samples]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data_t0[idx], self.data_t1[idx]

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
