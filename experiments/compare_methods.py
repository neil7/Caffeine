import subprocess
import json
import pandas as pd

# Run your Taylor method
subprocess.run([
    'python', 'main.py',
    '--dataset', 'cifar10',
    '--model_path', './ferrari_checkpoints/cifar10_global.pth',
    '--unlearn_ratio', '0.1',
    '--output_dir', './results/taylor'
])

# Run Ferrari baseline (from their code)
subprocess.run([
    'python', 'ferrari_main.py',
    '--dataset', 'cifar10',
    '--model_path', './ferrari_checkpoints/cifar10_global.pth',
    '--unlearn_ratio', '0.1',
    '--output_dir', './results/ferrari'
])

# Load and compare results
with open('./results/taylor/caffeine_unlearning_cifar10_client0.json') as f:
    taylor_results = json.load(f)

with open('./results/ferrari/ferrari_unlearning_cifar10_client0.json') as f:
    ferrari_results = json.load(f)

# Create comparison table
comparison = pd.DataFrame({
    'Method': ['Taylor (Ours)', 'Ferrari'],
    'Acc_unlearn_after': [
        taylor_results['accuracy']['unlearn_after'],
        ferrari_results['accuracy']['unlearn_after']
    ],
    'Acc_retain_after': [
        taylor_results['accuracy']['retain_after'],
        ferrari_results['accuracy']['retain_after']
    ],
    'Acc_test_after': [
        taylor_results['accuracy']['test_after'],
        ferrari_results['accuracy']['test_after']
    ]
})

print(comparison)
comparison.to_csv('./results/comparison.csv', index=False)
