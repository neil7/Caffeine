import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Tuple, List

class FederatedDataLoader:
    """
    Load and partition datasets for federated learning experiments.
    Compatible with Ferrari benchmark datasets.
    """
    
    def __init__(
        self,
        dataset_name: str,
        data_root: str = './data',
        num_clients: int = 10,
        batch_size: int = 32,
        iid: bool = True
    ):
        """
        Args:
            dataset_name: 'mnist', 'fashionmnist', 'cifar10', 'cifar100', 'celeba'
            data_root: Root directory for datasets
            num_clients: Number of federated clients
            batch_size: Batch size for DataLoaders
            iid: If True, distribute data IID; else non-IID
        """
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.iid = iid
        
        self.train_dataset, self.test_dataset = self._load_dataset()
        self.client_datasets = self._partition_data()
        
    def _load_dataset(self) -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
        """Load train and test datasets."""
        
        if self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(
                root=self.data_root, 
                train=True, 
                download=True, 
                transform=transform
            )
            test_dataset = datasets.MNIST(
                root=self.data_root, 
                train=False, 
                download=True, 
                transform=transform
            )
            
        elif self.dataset_name == 'fashionmnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = datasets.FashionMNIST(
                root=self.data_root, 
                train=True, 
                download=True, 
                transform=transform
            )
            test_dataset = datasets.FashionMNIST(
                root=self.data_root, 
                train=False, 
                download=True, 
                transform=transform
            )
            
        elif self.dataset_name == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
            train_dataset = datasets.CIFAR10(
                root=self.data_root, 
                train=True, 
                download=True, 
                transform=transform_train
            )
            test_dataset = datasets.CIFAR10(
                root=self.data_root, 
                train=False, 
                download=True, 
                transform=transform_test
            )
            
        elif self.dataset_name == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                   (0.2675, 0.2565, 0.2761))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), 
                                   (0.2675, 0.2565, 0.2761))
            ])
            train_dataset = datasets.CIFAR100(
                root=self.data_root, 
                train=True, 
                download=True, 
                transform=transform_train
            )
            test_dataset = datasets.CIFAR100(
                root=self.data_root, 
                train=False, 
                download=True, 
                transform=transform_test
            )
            
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        return train_dataset, test_dataset
    
    def _partition_data(self) -> List[Subset]:
        """Partition training data across clients."""
        
        if self.iid:
            return self._partition_iid()
        else:
            return self._partition_non_iid()
    
    def _partition_iid(self) -> List[Subset]:
        """IID data partitioning (random split)."""
        num_samples = len(self.train_dataset)
        indices = np.random.permutation(num_samples)
        
        # Split indices equally among clients
        split_size = num_samples // self.num_clients
        client_datasets = []
        
        for i in range(self.num_clients):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < self.num_clients - 1 else num_samples
            client_indices = indices[start_idx:end_idx]
            client_datasets.append(Subset(self.train_dataset, client_indices))
        
        return client_datasets
    
    def _partition_non_iid(self, num_shards: int = 2) -> List[Subset]:
        """
        Non-IID data partitioning (label distribution skew).
        Each client gets num_shards different label classes.
        """
        # Get labels
        if hasattr(self.train_dataset, 'targets'):
            labels = np.array(self.train_dataset.targets)
        elif hasattr(self.train_dataset, 'labels'):
            labels = np.array(self.train_dataset.labels)
        else:
            raise ValueError("Cannot extract labels from dataset")
        
        num_classes = len(np.unique(labels))
        num_samples = len(labels)
        
        # Sort indices by label
        sorted_indices = np.argsort(labels)
        
        # Create shards (200 samples each typically)
        shard_size = num_samples // (self.num_clients * num_shards)
        num_shards_total = self.num_clients * num_shards
        
        shards = []
        for i in range(num_shards_total):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < num_shards_total - 1 else num_samples
            shards.append(sorted_indices[start_idx:end_idx])
        
        # Shuffle shards and assign to clients
        np.random.shuffle(shards)
        
        client_datasets = []
        for i in range(self.num_clients):
            client_indices = np.concatenate(
                shards[i * num_shards:(i + 1) * num_shards]
            )
            client_datasets.append(Subset(self.train_dataset, client_indices))
        
        return client_datasets
    
    def get_client_loader(self, client_id: int) -> DataLoader:
        """Get DataLoader for specific client."""
        return DataLoader(
            self.client_datasets[client_id],
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def create_unlearn_retain_split(
        self,
        client_id: int,
        unlearn_ratio: float = 0.1
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Split client data into unlearn and retain sets.
        
        Args:
            client_id: Client index
            unlearn_ratio: Fraction of data to unlearn (0.0 to 1.0)
            
        Returns:
            (unlearn_loader, retain_loader)
        """
        client_dataset = self.client_datasets[client_id]
        num_samples = len(client_dataset)
        num_unlearn = int(num_samples * unlearn_ratio)
        
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        
        unlearn_indices = indices[:num_unlearn]
        retain_indices = indices[num_unlearn:]
        
        unlearn_dataset = Subset(client_dataset, unlearn_indices)
        retain_dataset = Subset(client_dataset, retain_indices)
        
        unlearn_loader = DataLoader(
            unlearn_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return unlearn_loader, retain_loader
