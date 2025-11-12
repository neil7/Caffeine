import torch
import torch.nn as nn
import argparse
from datasets.data_loader import FederatedDataLoader
from models.resnet import ResNet18
from models.cnn import SimpleCNN
from unlearning.caffeine_unlearning import CAFFEINE
from federated.server import FederatedServer
from federated.client import FederatedClient
import json
import os
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='CAFFEINE: Computationally-Efficient Federated Unlearning'
    )
    
    # Execution mode
    parser.add_argument('--mode', type=str, default='centralized',
                        choices=['centralized', 'federated'],
                        help='Execution mode: centralized or federated')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    
    # Federated Learning (only used in federated mode)
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of federated training rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Number of local training epochs per round')
    parser.add_argument('--iid', action='store_true',
                        help='IID data distribution')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    # Training (only for centralized mode or federated pre-training)
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='Training epochs for centralized mode')
    parser.add_argument('--train_model', action='store_true',
                        help='Train model from scratch (centralized mode)')
    
    # Unlearning
    parser.add_argument('--client_id', type=int, default=0,
                        help='Client performing unlearning')
    parser.add_argument('--unlearn_ratio', type=float, default=0.1,
                        help='Fraction of client data to unlearn')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for CAFFEINE approximation')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Unlearning strength')
    parser.add_argument('--num_iterations', type=int, default=3,
                        help='Number of unlearning iterations')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Perturbation magnitude for Taylor expansion')
    
    # Model
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint (optional)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cuda', 'mps', 'cpu', 'auto'],
                        help='Device to use (auto will select best available: cuda > mps > cpu)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    
    return parser.parse_args()


def load_or_create_model(model_path, dataset_name, num_classes, device):
    """Load existing model or create new one."""
    if dataset_name in ['mnist', 'fashionmnist']:
        model = SimpleCNN(num_classes=num_classes)
    else:
        input_channels = 3
        model = ResNet18(num_classes=num_classes, input_channels=input_channels)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Creating new model...")
    
    model = model.to(device)
    return model


def train_model_centralized(model, train_loader, device, epochs=10, lr=0.001):
    """Train model from scratch in centralized setting."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\n" + "="*60)
    print("TRAINING MODEL FROM SCRATCH")
    print("="*60)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    
    return model


def federated_training(server, clients, num_rounds, local_epochs, device):
    """Run federated learning for specified rounds."""
    print("\n" + "="*60)
    print("FEDERATED TRAINING")
    print("="*60)
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        # Get global parameters
        global_params = server.get_global_params()
        
        # Each client updates with global model
        for client in clients:
            client.update_model(global_params)
        
        # Each client performs local training
        client_updates = []
        for i, client in enumerate(clients):
            if (round_num * len(clients) + i) % 10 == 0:
                print(f"  Client {i} training...")
            updated_params = client.local_training(
                epochs=local_epochs,
                lr=0.01
            )
            client_updates.append(updated_params)
        
        # Server aggregates updates (FedAvg)
        aggregated_params = server.aggregate(client_updates)
        server.update_global_model(aggregated_params)
        
        if (round_num + 1) % 10 == 0 or round_num == 0:
            print(f"Round {round_num + 1} completed")


def run_centralized_mode(args, device):
    """Run centralized unlearning experiment."""
    print("\n" + "="*70)
    print("PHASE 1: CENTRALIZED MODE")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    from torchvision import datasets, transforms
    from torch.utils.data import random_split
    
    # Get dataset with transforms
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(
            root=args.data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=args.data_root, train=False, download=True, transform=transform
        )
        num_classes = 10
    elif args.dataset == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(
            root=args.data_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=args.data_root, train=False, download=True, transform=transform
        )
        num_classes = 10
    elif args.dataset == 'cifar10':
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
            root=args.data_root, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_root, train=False, download=True, transform=transform_test
        )
        num_classes = 10
    else:  # cifar100
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
            root=args.data_root, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_root, train=False, download=True, transform=transform_test
        )
        num_classes = 100
    
    # Create model
    model = load_or_create_model(args.model_path, args.dataset, num_classes, device)
    
    # Train model if requested
    if args.train_model or args.model_path is None:
        full_train_loader = DataLoader(
            train_dataset, 
            batch_size=128, 
            shuffle=True
        )
        model = train_model_centralized(
            model, 
            full_train_loader, 
            device, 
            epochs=args.train_epochs
        )
        
        # Save trained model
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict()
        }, os.path.join(args.output_dir, f'{args.dataset}_trained_model.pth'))
    
    # Split into retain and unlearn sets
    train_size = int(len(train_dataset) * (1 - args.unlearn_ratio))
    forget_size = len(train_dataset) - train_size
    train_subset, forget_subset = random_split(
        train_dataset, 
        [train_size, forget_size]
    )
    
    # Create data loaders
    retain_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
    unlearn_loader = DataLoader(forget_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Run CAFFEINE unlearning
    return run_caffeine_unlearning(
        model, unlearn_loader, retain_loader, test_loader, 
        args, device, mode='centralized'
    )


def run_federated_mode(args, device):
    """Run federated unlearning experiment."""
    print("\n" + "="*70)
    print("PHASE 2: FEDERATED MODE")
    print("="*70)
    
    # Load federated dataset
    print(f"\nLoading {args.dataset} dataset for {args.num_clients} clients...")
    fed_data = FederatedDataLoader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        iid=args.iid
    )
    
    # Determine number of classes
    num_classes_map = {
        'mnist': 10,
        'fashionmnist': 10,
        'cifar10': 10,
        'cifar100': 100
    }
    num_classes = num_classes_map[args.dataset]
    
    # Initialize model
    global_model = load_or_create_model(args.model_path, args.dataset, num_classes, device)
    
    # Initialize server
    server = FederatedServer(global_model, device)
    
    # Initialize clients
    clients = []
    for i in range(args.num_clients):
        client_loader = fed_data.get_client_loader(i)
        client = FederatedClient(
            client_id=i,
            model=global_model,
            train_loader=client_loader,
            device=device
        )
        clients.append(client)
    
    # Federated training (if no pre-trained model)
    if args.model_path is None or args.train_model:
        federated_training(
            server=server,
            clients=clients,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            device=device
        )
        
        # Save trained global model
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({
            'model_state_dict': server.global_model.state_dict()
        }, os.path.join(args.output_dir, f'{args.dataset}_federated_trained_model.pth'))
    
    # Evaluate global model before unlearning
    test_loader = fed_data.get_test_loader()
    acc_before = evaluate_model(server.global_model, test_loader, device)
    print(f"\nGlobal model test accuracy before unlearning: {acc_before:.2f}%")
    
    # Client unlearning with CAFFEINE
    print("\n" + "="*60)
    print(f"CLIENT {args.client_id} PERFORMING UNLEARNING WITH CAFFEINE")
    print("="*60)
    
    # Get unlearn/retain split for client
    unlearn_loader, retain_loader = fed_data.create_unlearn_retain_split(
        client_id=args.client_id,
        unlearn_ratio=args.unlearn_ratio
    )
    
    # Client updates with global model
    clients[args.client_id].update_model(server.get_global_params())
    
    # Run CAFFEINE unlearning
    unlearned_model, metrics = run_caffeine_unlearning(
        clients[args.client_id].model,
        unlearn_loader,
        retain_loader,
        test_loader,
        args,
        device,
        mode='federated'
    )
    
    # Update global model (in practice, might aggregate with other clients)
    server.global_model = unlearned_model
    
    # Evaluate after unlearning
    acc_after = evaluate_model(server.global_model, test_loader, device)
    print(f"\nGlobal model test accuracy after unlearning: {acc_after:.2f}%")
    
    # Save results
    metrics['global_acc_before'] = acc_before
    metrics['global_acc_after'] = acc_after
    
    return unlearned_model, metrics


def evaluate_model(model, data_loader, device):
    """Compute accuracy on data loader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


def run_caffeine_unlearning(model, unlearn_loader, retain_loader, test_loader, args, device, mode='centralized'):
    """Run CAFFEINE unlearning algorithm."""
    
    # Initialize CAFFEINE
    print(f"\nInitializing CAFFEINE unlearning...")
    caffeine = CAFFEINE(
        model=model,
        device=device,
        learning_rate=args.lr,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Compute initial metrics
    print("\n" + "="*60)
    print("BEFORE UNLEARNING")
    print("="*60)
    acc_unlearn_before = caffeine.compute_accuracy(unlearn_loader)
    acc_retain_before = caffeine.compute_accuracy(retain_loader)
    acc_test_before = caffeine.compute_accuracy(test_loader)
    
    print(f"Unlearn set accuracy: {acc_unlearn_before:.2f}%")
    print(f"Retain set accuracy:  {acc_retain_before:.2f}%")
    print(f"Test set accuracy:    {acc_test_before:.2f}%")
    
    # Perform unlearning
    print("\n" + "="*60)
    print("PERFORMING CAFFEINE UNLEARNING")
    print("="*60)
    
    updated_model, unlearn_metrics = caffeine.unlearn(
        unlearn_loader=unlearn_loader,
        retain_loader=retain_loader,
        criterion=criterion,
        alpha=args.alpha
    )
    
    # Compute final metrics
    print("\n" + "="*60)
    print("AFTER UNLEARNING")
    print("="*60)
    acc_unlearn_after = caffeine.compute_accuracy(unlearn_loader)
    acc_retain_after = caffeine.compute_accuracy(retain_loader)
    acc_test_after = caffeine.compute_accuracy(test_loader)
    
    print(f"Unlearn set accuracy: {acc_unlearn_after:.2f}%")
    print(f"Retain set accuracy:  {acc_retain_after:.2f}%")
    print(f"Test set accuracy:    {acc_test_after:.2f}%")
    
    # Compile results
    results = {
        'mode': mode,
        'args': vars(args),
        'accuracy': {
            'unlearn_before': acc_unlearn_before,
            'unlearn_after': acc_unlearn_after,
            'retain_before': acc_retain_before,
            'retain_after': acc_retain_after,
            'test_before': acc_test_before,
            'test_after': acc_test_after
        },
        'metrics': {
            'unlearn_loss_before': unlearn_metrics['unlearn_loss_before'],
            'unlearn_loss_after': unlearn_metrics['unlearn_loss_after'],
            'retain_loss_before': unlearn_metrics['retain_loss_before'],
            'retain_loss_after': unlearn_metrics['retain_loss_after'],
            'influence_norms': unlearn_metrics['influence_norm']
        }
    }
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = os.path.join(
        args.output_dir,
        f'caffeine_{mode}_{args.dataset}_client{args.client_id}.json'
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_file}")
    
    # Save updated model
    model_output = os.path.join(
        args.output_dir,
        f'caffeine_{mode}_{args.dataset}_client{args.client_id}.pth'
    )
    torch.save({
        'model_state_dict': updated_model.state_dict(),
        'args': vars(args),
        'results': results
    }, model_output)
    
    print(f"Updated model saved to {model_output}")
    
    return updated_model, results


def get_device(device_arg):
    """
    Intelligently select the best available device.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            print(f"Auto-detected: NVIDIA GPU ({device_name})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Auto-detected: Apple Silicon (MPS)")
        else:
            device = torch.device('cpu')
            print(f"Auto-detected: CPU")
    else:
        # Manual device selection with validation
        if device_arg == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_name = torch.cuda.get_device_name(0)
                print(f"Using: NVIDIA GPU ({device_name})")
            else:
                print("Warning: CUDA not available, falling back to CPU")
                device = torch.device('cpu')
        elif device_arg == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                print(f"Using: Apple Silicon (MPS)")
            else:
                print("Warning: MPS not available, falling back to CPU")
                device = torch.device('cpu')
        else:  # cpu
            device = torch.device('cpu')
            print(f"Using: CPU")
    
    return device


def main():
    args = parse_args()
    
    # Set device with intelligent selection
    print(f"\n{'='*70}")
    print(f"CAFFEINE: Computationally-Efficient Federated Unlearning")
    print(f"{'='*70}")
    device = get_device(args.device)
    print(f"Mode: {args.mode.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run appropriate mode
    if args.mode == 'centralized':
        model, results = run_centralized_mode(args, device)
    else:  # federated
        model, results = run_federated_mode(args, device)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nFinal Test Accuracy: {results['accuracy']['test_after']:.2f}%")
    print(f"Unlearn Effectiveness: {results['accuracy']['unlearn_before']:.2f}% → {results['accuracy']['unlearn_after']:.2f}%")
    print(f"Retain Preservation: {results['accuracy']['retain_before']:.2f}% → {results['accuracy']['retain_after']:.2f}%")


if __name__ == '__main__':
    main()
