import torch
import torch.nn as nn
from typing import Dict, List
import copy

class FederatedClient:
    """
    Simulates a client in federated learning.
    Each client trains on local data and can perform local unlearning.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: str = 'cuda'  # 'cuda', 'mps', or 'cpu'
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        
    def local_training(
        self,
        epochs: int = 5,
        lr: float = 0.001
    ) -> Dict[str, torch.Tensor]:
        """
        Perform local training on client data.
        
        Returns:
            Dictionary of updated model parameters
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Return updated parameters
        return {
            name: param.data.clone() 
            for name, param in self.model.named_parameters()
        }
    
    def update_model(self, global_params: Dict[str, torch.Tensor]):
        """Update local model with global parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = global_params[name].clone()
    
    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
