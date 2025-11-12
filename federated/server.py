import torch
import torch.nn as nn
from typing import Dict, List
import copy

class FederatedServer:
    """
    Federated learning server that aggregates client updates.
    Implements FedAvg algorithm.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'  # 'cuda', 'mps', or 'cpu'
    ):
        self.global_model = model.to(device)
        self.device = device
        
    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        client_weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg aggregation: weighted average of client parameters.
        
        Args:
            client_params: List of client parameter dictionaries
            client_weights: Weight for each client (default: equal weights)
            
        Returns:
            Aggregated global parameters
        """
        num_clients = len(client_params)
        
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        for name in client_params[0].keys():
            aggregated_params[name] = torch.zeros_like(
                client_params[0][name]
            ).to(self.device)
            
            # Weighted sum
            for client_param, weight in zip(client_params, client_weights):
                aggregated_params[name] += weight * client_param[name].to(self.device)
        
        return aggregated_params
    
    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """Update global model with aggregated parameters."""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data = aggregated_params[name].clone()
    
    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters."""
        return {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
        }
