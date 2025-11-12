import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import copy



class CAFFEINE:
    """
    CAFFEINE: Computationally-Efficient Federated Unlearning
    via First-Order Influence Estimation
    
    Hessian-free federated unlearning using Taylor expansion.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        num_iterations: int = 1,
        epsilon: float = 0.01  # Perturbation magnitude
    ):
        """
        Args:
            model: The trained global model
            device: 'cuda', 'mps', or 'cpu' (GPU or CPU device)
            learning_rate: Step size for gradient perturbation
            num_iterations: Number of unlearning iterations
            epsilon: Magnitude for parameter perturbation
        """
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        
    def compute_gradient(
        self, 
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient of loss w.r.t. model parameters.
        
        Args:
            data_loader: DataLoader for unlearning data
            criterion: Loss function
            
        Returns:
            Dictionary mapping parameter names to gradients
        """
        self.model.eval()
        self.model.zero_grad()
        
        total_loss = 0.0
        num_samples = 0
        
        # Accumulate gradients over the dataset
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * data.size(0)
            num_samples += data.size(0)
        
        # Extract gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone() / num_samples
        
        return gradients
    
    def perturb_model(
        self, 
        gradients: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """
        Create perturbed model θ' = θ - ε * ∇L(θ, D_u)
        
        Args:
            gradients: Gradient dictionary from compute_gradient
            
        Returns:
            Perturbed model copy
        """
        perturbed_model = copy.deepcopy(self.model)
        
        with torch.no_grad():
            for name, param in perturbed_model.named_parameters():
                if name in gradients:
                    param.data -= self.lr * gradients[name]
        
        return perturbed_model
    
    def compute_influence_approximation(
        self,
        unlearn_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Compute influence approximation using Taylor expansion:
        Δθ ≈ ∇L(θ, D_u) - ∇L(θ', D_u)
        
        This avoids Hessian computation entirely.
        
        Args:
            unlearn_loader: DataLoader for data to unlearn
            criterion: Loss function
            
        Returns:
            Influence approximation (parameter update direction)
        """
        # Step 1: Compute gradient at current parameters θ
        grad_theta = self.compute_gradient(unlearn_loader, criterion)
        
        # Step 2: Perturb model to get θ'
        perturbed_model = self.perturb_model(grad_theta)
        
        # Step 3: Compute gradient at perturbed parameters θ'
        original_model = self.model
        self.model = perturbed_model
        grad_theta_prime = self.compute_gradient(unlearn_loader, criterion)
        self.model = original_model  # Restore
        
        # Step 4: Compute influence = grad_theta - grad_theta_prime
        influence = {}
        for name in grad_theta.keys():
            influence[name] = grad_theta[name] - grad_theta_prime[name]
        
        return influence
    
    def unlearn(
        self,
        unlearn_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        alpha: float = 0.1  # Unlearning strength
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Perform federated unlearning on client data.
        
        Args:
            unlearn_loader: Data to forget
            retain_loader: Data to retain (for validation)
            criterion: Loss function
            alpha: Step size for applying influence
            
        Returns:
            Updated model and metrics dictionary
        """
        metrics = {
            'unlearn_loss_before': [],
            'unlearn_loss_after': [],
            'retain_loss_before': [],
            'retain_loss_after': [],
            'influence_norm': []
        }
        
        # Compute initial losses
        metrics['unlearn_loss_before'] = self._compute_loss(unlearn_loader, criterion)
        metrics['retain_loss_before'] = self._compute_loss(retain_loader, criterion)
        
        for iteration in range(self.num_iterations):
            # Compute influence approximation
            influence = self.compute_influence_approximation(
                unlearn_loader,
                criterion
            )
            
            # Compute influence magnitude
            influence_norm = sum(
                torch.norm(grad).item() for grad in influence.values()
            )
            metrics['influence_norm'].append(influence_norm)
            
            # Apply anti-update: θ_new = θ + α * influence
            # (We add because influence is negative gradient direction)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in influence:
                        param.data += alpha * influence[name]
            
            print(f"Iteration {iteration + 1}/{self.num_iterations}, "
                  f"Influence norm: {influence_norm:.4f}")
        
        # Compute final losses
        metrics['unlearn_loss_after'] = self._compute_loss(unlearn_loader, criterion)
        metrics['retain_loss_after'] = self._compute_loss(retain_loader, criterion)
        
        print(f"\nUnlearning completed:")
        print(f"  Unlearn loss: {metrics['unlearn_loss_before']:.4f} → "
              f"{metrics['unlearn_loss_after']:.4f}")
        print(f"  Retain loss: {metrics['retain_loss_before']:.4f} → "
              f"{metrics['retain_loss_after']:.4f}")
        
        return self.model, metrics
    
    def _compute_loss(
        self, 
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> float:
        """Helper function to compute average loss on a dataset."""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
        
        return total_loss / num_samples if num_samples > 0 else 0.0

    def compute_accuracy(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> float:
        """Compute classification accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total if total > 0 else 0.0
