import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch.utils.data.sampler import (
    BatchSampler,
    SubsetRandomSampler,
    SequentialSampler,
)

class CustomDataLoader(DataLoader):
    def __init__(
        self, data_container, batch_size, indices, shuffle, seed=None, **kwargs
    ):

        if shuffle:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)
            idx_sampler = SubsetRandomSampler(indices, generator)
        else:
            idx_sampler = SequentialSampler(Subset(data_container, indices))

        batch_sampler = BatchSampler(
            idx_sampler, batch_size=batch_size, drop_last=False
        )

        super().__init__(
            data_container,
            sampler=batch_sampler,
            collate_fn=data_container.collate_fn,
            pin_memory=True,  # load on CPU push to GPU
            **kwargs
        )


class DataProvider:
    """
    Parameters
    ----------
        data_container: DataContainer
            Contains the dataset.
        ntrain: int
            Number of samples in the training set.
        nval: int
            Number of samples in the validation set.
        batch_size: int
            Number of samples to process at once.
        seed: int
            Seed for drawing samples into train and val set (and shuffle).
        random_split: bool
            If True put the samples randomly into the subsets else in order.
        shuffle: bool
            If True shuffle the samples after each epoch.
        sample_with_replacement: bool
            Sample data from the dataset with replacement.
        transforms: list
            List of transformations applied to the dataset.
    """

    def __init__(
        self,
        data_container,
        ntrain: int,
        nval: int,
        batch_size: int = 1,
        seed: int = None,
        random_split: bool = False,
        shuffle: bool = True,
        sample_with_replacement: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        self.data_container = data_container
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.kwargs = kwargs

        # Random state parameter, such that random operations are reproducible if wanted
        _random_state = np.random.RandomState(seed=seed)

        all_idx = np.arange(len(data_container))
        if random_split:
            # Shuffle indices
            all_idx = _random_state.permutation(all_idx)

        if sample_with_replacement:
            # Sample with replacement so as to train an ensemble of Dimenets
            all_idx = _random_state.choice(all_idx, len(all_idx), replace=True)

        # Store indices of training, validation and test data
        self.idx = {
            "train": all_idx[0:ntrain],
            "val": all_idx[ntrain : ntrain + nval],
            "test": all_idx[ntrain + nval :],
        }

    def get_dataset(self, split, batch_size=None):
        assert split in self.idx
        if batch_size is None:
            batch_size = self.batch_size
        shuffle = self.shuffle if split == "train" else False

        dataloader = CustomDataLoader(
            self.data_container,
            batch_size=batch_size,
            indices=self.idx[split],
            shuffle=shuffle,
            **self.kwargs
        )

        # loop infinitely
        # we use the generator as the rest of the code is based on steps and not epochs
        def generator():
            while True:
                for inputs, targets in dataloader:
                    yield inputs, targets

        return generator()

    def get_loader(self, split, batch_size=None):
        assert split in self.idx
        if batch_size is None:
            batch_size = self.batch_size
        shuffle = self.shuffle if split == "train" else False
        return CustomDataLoader(self.data_container, batch_size=batch_size,
            indices=self.idx[split], shuffle=shuffle, **self.kwargs)

    def get_distributed_loader(self, split, world_size, rank, batch_size=None):
        dataset = Subset(self.data_container, self.idx[split])
        batch_size = batch_size if batch_size is not None else self.batch_size
        sampler = DistributedSampler(dataset, world_size, rank)
        loader = DataLoader(dataset, self.batch_size, sampler=sampler, num_workers=self.num_workers,
                                collate_fn=self.data_container.collate_fn, pin_memory=True,)
        return loader