"""
Federated Dataset Utilities
Handles data partitioning for federated learning scenarios (IID and non-IID).
"""

import torch
from torch.utils.data import Subset
import numpy as np
from typing import List, Tuple


class FederatedDataset:
    """
    Partition dataset across federated clients with IID or non-IID distribution.
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_clients: int = 10,
        iid: bool = True,
        num_shards_per_client: int = 2
    ):
        """
        Args:
            dataset: PyTorch dataset to partition
            num_clients: Number of federated clients
            iid: If True, distribute data IID; else non-IID
            num_shards_per_client: For non-IID, number of shards per client
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.iid = iid
        self.num_shards_per_client = num_shards_per_client
        
        self.client_datasets = self._partition_data()
        
    def _partition_data(self) -> List[Subset]:
        """Partition data based on IID or non-IID setting."""
        if self.iid:
            return self._partition_iid()
        else:
            return self._partition_non_iid()
    
    def _partition_iid(self) -> List[Subset]:
        """
        IID partitioning: randomly shuffle and split data equally.
        """
        num_samples = len(self.dataset)
        indices = np.random.permutation(num_samples)
        
        # Split indices equally among clients
        split_size = num_samples // self.num_clients
        client_datasets = []
        
        for i in range(self.num_clients):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < self.num_clients - 1 else num_samples
            client_indices = indices[start_idx:end_idx]
            client_datasets.append(Subset(self.dataset, client_indices))
        
        print(f"IID partitioning: {self.num_clients} clients, "
              f"~{split_size} samples per client")
        
        return client_datasets
    
    def _partition_non_iid(self) -> List[Subset]:
        """
        Non-IID partitioning: label distribution skew.
        Each client gets `num_shards_per_client` shards of sorted data.
        
        This follows the approach from McMahan et al. (2017) and is used
        in Ferrari for non-IID experiments.
        """
        # Extract labels
        if hasattr(self.dataset, 'targets'):
            labels = np.array(self.dataset.targets)
        elif hasattr(self.dataset, 'labels'):
            labels = np.array(self.dataset.labels)
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute")
        
        num_samples = len(labels)
        num_classes = len(np.unique(labels))
        
        # Sort indices by label
        sorted_indices = np.argsort(labels)
        
        # Create shards (each shard has samples from similar labels)
        num_shards = self.num_clients * self.num_shards_per_client
        shard_size = num_samples // num_shards
        
        shards = []
        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = (i + 1) * shard_size if i < num_shards - 1 else num_samples
            shards.append(sorted_indices[start_idx:end_idx])
        
        # Randomly shuffle shards
        np.random.shuffle(shards)
        
        # Assign shards to clients
        client_datasets = []
        for i in range(self.num_clients):
            client_shards = shards[
                i * self.num_shards_per_client:(i + 1) * self.num_shards_per_client
            ]
            client_indices = np.concatenate(client_shards)
            client_datasets.append(Subset(self.dataset, client_indices))
        
        print(f"Non-IID partitioning: {self.num_clients} clients, "
              f"{self.num_shards_per_client} shards per client")
        
        return client_datasets
    
    def get_client_dataset(self, client_id: int) -> Subset:
        """Get dataset for specific client."""
        if client_id >= self.num_clients:
            raise ValueError(f"client_id {client_id} >= num_clients {self.num_clients}")
        return self.client_datasets[client_id]
    
    def get_all_client_datasets(self) -> List[Subset]:
        """Get all client datasets."""
        return self.client_datasets
    
    def split_client_data(
        self,
        client_id: int,
        split_ratio: float = 0.1,
        shuffle: bool = True
    ) -> Tuple[Subset, Subset]:
        """
        Split a client's data into two subsets (e.g., unlearn and retain).
        
        Args:
            client_id: Client index
            split_ratio: Fraction for first subset (e.g., unlearn set)
            shuffle: Whether to shuffle before splitting
            
        Returns:
            (subset1, subset2) where subset1 has split_ratio of data
        """
        client_dataset = self.client_datasets[client_id]
        num_samples = len(client_dataset)
        
        indices = list(range(num_samples))
        if shuffle:
            np.random.shuffle(indices)
        
        split_point = int(num_samples * split_ratio)
        indices_1 = indices[:split_point]
        indices_2 = indices[split_point:]
        
        subset_1 = Subset(client_dataset, indices_1)
        subset_2 = Subset(client_dataset, indices_2)
        
        return subset_1, subset_2
    
    def get_label_distribution(self, client_id: int) -> dict:
        """
        Get label distribution for a specific client (useful for debugging).
        
        Returns:
            Dictionary mapping label to count
        """
        client_dataset = self.client_datasets[client_id]
        
        # Get labels for this client's data
        labels = []
        for idx in client_dataset.indices:
            if hasattr(self.dataset, 'targets'):
                label = self.dataset.targets[idx]
            elif hasattr(self.dataset, 'labels'):
                label = self.dataset.labels[idx]
            else:
                raise ValueError("Cannot extract labels")
            labels.append(label)
        
        # Count occurrences
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def print_statistics(self):
        """Print statistics about the federated partitioning."""
        print("\n" + "="*60)
        print("FEDERATED DATASET STATISTICS")
        print("="*60)
        print(f"Total samples: {len(self.dataset)}")
        print(f"Number of clients: {self.num_clients}")
        print(f"Distribution: {'IID' if self.iid else 'Non-IID'}")
        print(f"\nSamples per client:")
        
        for i in range(min(self.num_clients, 10)):  # Show first 10 clients
            client_dataset = self.client_datasets[i]
            label_dist = self.get_label_distribution(i)
            print(f"  Client {i}: {len(client_dataset)} samples, "
                  f"Labels: {label_dist}")
        
        if self.num_clients > 10:
            print(f"  ... ({self.num_clients - 10} more clients)")
        print("="*60 + "\n")


def create_federated_split(
    dataset: torch.utils.data.Dataset,
    num_clients: int = 10,
    iid: bool = True,
    alpha: float = 0.5
) -> List[Subset]:
    """
    Convenience function to create federated split.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        iid: IID or non-IID distribution
        alpha: Dirichlet concentration parameter for non-IID (lower = more skew)
        
    Returns:
        List of client datasets
    """
    fed_dataset = FederatedDataset(
        dataset=dataset,
        num_clients=num_clients,
        iid=iid
    )
    
    return fed_dataset.get_all_client_datasets()


def dirichlet_non_iid_split(
    dataset: torch.utils.data.Dataset,
    num_clients: int = 10,
    alpha: float = 0.5
) -> List[Subset]:
    """
    Advanced non-IID split using Dirichlet distribution.
    More flexible than shard-based approach.
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration (lower = more heterogeneous)
        
    Returns:
        List of client datasets
    """
    # Extract labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels'")
    
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Sample proportions from Dirichlet distribution
    client_class_proportions = np.random.dirichlet(
        [alpha] * num_clients, 
        num_classes
    )
    
    # Assign samples to clients based on proportions
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = client_class_proportions[c]
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)
        
        start = 0
        for client_id, end in enumerate(split_points):
            client_indices[client_id].extend(class_indices[start:end])
            start = end
    
    # Create client datasets
    client_datasets = [
        Subset(dataset, np.array(indices)) 
        for indices in client_indices
    ]
    
    print(f"Dirichlet non-IID split: {num_clients} clients, alpha={alpha}")
    
    return client_datasets
