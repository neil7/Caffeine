Here's a crisp and professional README.md for your CAFFEINE implementation:

```markdown
# CAFFEINE: Computationally-Efficient Federated Unlearning

**CAFFEINE** (Computationally-Efficient Federated Unlearning via First-Order Influence Estimation) is a Hessian-free federated unlearning framework that uses Taylor expansion to approximate influence functions without expensive second-order computations.

## 🚀 Key Features

- **Hessian-Free**: Replaces \(O(d^3)\) Hessian inversion with \(O(d)\) gradient computations
- **Taylor Expansion**: First-order approximation: `Δθ ≈ ∇L(θ) - ∇L(θ')`
- **Dual Modes**: Supports both centralized and federated learning scenarios
- **Privacy-Preserving**: Local unlearning without requiring other clients' participation
- **Benchmark Ready**: Direct comparison with Ferrari (NeurIPS 2024)

## 📦 Installation

```
# Clone repository
git clone https://github.com/yourusername/caffeine-unlearning.git
cd caffeine-unlearning

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy, pandas, matplotlib

**GPU Support:**
- NVIDIA GPUs (CUDA)
- Apple Silicon (M1/M2/M3) via MPS (Metal Performance Shaders)
- Automatic device detection

## 🎯 Quick Start

### Centralized Mode (Phase 1)

```
# Train and unlearn on MNIST
python main.py \
    --mode centralized \
    --dataset mnist \
    --train_model \
    --unlearn_ratio 0.1 \
    --output_dir ./results/centralized
```

### Federated Mode (Phase 2)

```
# Federated learning with 10 clients
python main.py \
    --mode federated \
    --dataset cifar10 \
    --num_clients 10 \
    --num_rounds 50 \
    --train_model \
    --client_id 0 \
    --unlearn_ratio 0.1 \
    --output_dir ./results/federated
```

### GPU Acceleration

```bash
# Auto-detect best device (recommended)
python main.py --mode centralized --dataset cifar10 --device auto

# Force NVIDIA GPU
python main.py --mode centralized --dataset cifar10 --device cuda

# Force Apple Silicon
python main.py --mode centralized --dataset cifar10 --device mps

# Force CPU
python main.py --mode centralized --dataset cifar10 --device cpu
```

## 📁 Project Structure

```
caffeine-unlearning/
├── data/                      # Dataset storage
├── datasets/                  # Data loading utilities
│   ├── data_loader.py
│   └── federated_dataset.py
├── models/                    # Model architectures
│   ├── cnn.py
│   └── resnet.py
├── unlearning/               # Core CAFFEINE algorithm
│   └── caffeine_unlearning.py
├── federated/                # FL components
│   ├── server.py
│   └── client.py
├── main.py                   # Unified entry point
└── requirements.txt
```

## 🔧 Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | `centralized` or `federated` | `centralized` |
| `--dataset` | Dataset name | `cifar10` |
| `--unlearn_ratio` | Fraction of data to forget | `0.1` |
| `--alpha` | Unlearning strength | `0.1` |
| `--num_iterations` | CAFFEINE iterations | `3` |
| `--lr` | Learning rate for Taylor approximation | `0.001` |
| `--num_clients` | Number of FL clients | `10` |
| `--num_rounds` | FL training rounds | `50` |
| `--device` | Device: `auto`, `cuda`, `mps`, `cpu` | `auto` |

## 📊 Supported Datasets

- MNIST (28×28 grayscale, 10 classes)
- Fashion-MNIST (28×28 grayscale, 10 classes)
- CIFAR-10 (32×32 RGB, 10 classes)
- CIFAR-100 (32×32 RGB, 100 classes)

## 🧪 Evaluation Metrics

CAFFEINE tracks:
- **Unlearn Effectiveness**: Accuracy drop on forgotten data
- **Retain Preservation**: Accuracy maintenance on retained data
- **Test Utility**: Overall model performance
- **Computational Cost**: Runtime and FLOPs comparison

## 📖 Method Overview

CAFFEINE approximates the influence of removing data point \((x_i, y_i)\) via:

```
Influence ≈ ∇L(θ, D_u) - ∇L(θ', D_u)
```

where:
- `θ` = current model parameters
- `θ' = θ - ε∇L(θ, D_u)` = perturbed parameters
- `D_u` = data to unlearn

**Advantages over Hessian-based methods:**
- No matrix inversion required
- Minimal memory footprint
- Scalable to large models

## 🏆 Benchmarking

Compare against Ferrari baseline:

```
# Run CAFFEINE
python main.py --mode federated --dataset cifar10 --output_dir ./results/caffeine

# Run Ferrari (from their repo)
python ferrari_main.py --dataset cifar10 --output_dir ./results/ferrari

# Compare results
python compare_results.py --caffeine ./results/caffeine --ferrari ./results/ferrari
```

## 📄 Citation

If you use CAFFEINE in your research, please cite:

```
@inproceedings{sharma2025caffeine,
  title={CAFFEINE: Computationally-Efficient Federated Unlearning via First-Order Influence Estimation},
  author={Sharma, Neil and [Your Name]},
  booktitle={Proceedings of Middleware Conference},
  year={2025}
}
```

**Benchmark Reference:**
```
@inproceedings{gu2024ferrari,
  title={Ferrari: Federated Feature Unlearning via Optimizing Feature Sensitivity},
  author={Gu, Hanlin and Ong, Win Kent and Chan, Chee Seng and Fan, Lixin},
  booktitle={NeurIPS},
  year={2024}
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Ferrari implementation: [Federated-Feature-Unlearning](https://github.com/OngWinKent/Federated-Feature-Unlearning)
- Computational efficiency approach based on doctoral symposium paper (Middleware 2025)
- Benchmark datasets: MNIST, CIFAR-10/100

## 📧 Contact

For questions or collaborations:
- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/caffeine-unlearning/issues)
- Email: your.email@university.edu

---

**Status:** 🚧 Active Development | **Version:** 0.1.0 | **Last Updated:** November 2025