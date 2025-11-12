import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models.cnn import SimpleCNN
from models.resnet import ResNet18
from unlearning.caffeine_unlearning import CAFFEINE
import argparse

def train_model(model, train_loader, device, epochs=10):
    """Train a model from scratch."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {acc:.2f}%")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        model = SimpleCNN(num_classes=10)
    else:  # cifar10
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
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        model = ResNet18(num_classes=10)
    
    # Split training data into train + forget
    train_size = int(len(train_dataset) * (1 - args.unlearn_ratio))
    forget_size = len(train_dataset) - train_size
    train_subset, forget_subset = random_split(
        train_dataset, 
        [train_size, forget_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    forget_loader = DataLoader(forget_subset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Train model on full training set (including forget set)
    print("\n" + "="*60)
    print("PHASE 1: TRAINING ORIGINAL MODEL")
    print("="*60)
    full_train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model = train_model(model, full_train_loader, device, epochs=10)
    
    # Save trained model
    torch.save({
        'model_state_dict': model.state_dict()
    }, f'./results/{args.dataset}_trained_model.pth')
    
    # Initialize CAFFEINE unlearning
    print("\n" + "="*60)
    print("PHASE 2: APPLYING CAFFEINE UNLEARNING")
    print("="*60)
    
    caffeine = CAFFEINE(
        model=model,
        device=device,
        learning_rate=0.001,
        num_iterations=3
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Perform unlearning
    updated_model, metrics = caffeine.unlearn(
        unlearn_loader=forget_loader,
        retain_loader=train_loader,
        criterion=criterion,
        alpha=0.1
    )
    
    # Evaluate final model
    print("\n" + "="*60)
    print("PHASE 3: EVALUATION")
    print("="*60)
    
    acc_forget = caffeine.compute_accuracy(forget_loader)
    acc_retain = caffeine.compute_accuracy(train_loader)
    acc_test = caffeine.compute_accuracy(test_loader)
    
    print(f"\nFinal Results:")
    print(f"  Forget set accuracy: {acc_forget:.2f}%")
    print(f"  Retain set accuracy: {acc_retain:.2f}%")
    print(f"  Test set accuracy:   {acc_test:.2f}%")
    
    # Save unlearned model
    torch.save({
        'model_state_dict': updated_model.state_dict(),
        'metrics': metrics
    }, f'./results/{args.dataset}_unlearned_model.pth')

if __name__ == '__main__':
    main()
