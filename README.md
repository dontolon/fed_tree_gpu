# FedTree  
*A lightweight federated learning framework for visual classification.*

FedTree is a modular and easy-to-extend prototype for **federated learning (FL)** experiments.  
It supports multiple aggregation algorithms, centralized baselines, and datasets like **CIFAR-100** and **iNaturalist**.  


---

##  Features

- **Federated learning algorithms**
  - **FedAvg** – standard parameter averaging  
  - **FedProx** – FedAvg with proximal term for heterogeneous clients  
  - **FedNova** – update normalization by local steps  
  - **FedAvgM** – FedAvg with server-side momentum  

- **Centralized baseline** for direct performance comparison  
- **Non-IID data partitioning** using a Dirichlet distribution (`--alpha`)  
- **Backbones:** MobileNetV3-Small and ResNet18 (easily extendable)  
- **Evaluation metrics:** accuracy, per-class accuracy, precision, recall, F1, confusion matrix  
- **Datasets supported:**
  - CIFAR-100 (tree subset or random subset)
  - iNaturalist 2021 (mini) filtered for *Plantae* species  

---

##  Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

(Optional, for GPU acceleration)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### 2. Run a federated experiment (CIFAR-100)

Train with **FedAvg**:
```bash
python3 train.py --dataset cifar100 --agg fedavg --rounds 30 \
    --num_clients 4 --local_epochs 5 --output_dir ./outputs/fedavg_example
```

Train with **FedProx**:
```bash
python3 train.py --dataset cifar100 --agg fedprox --mu 0.01 --rounds 50 \
    --num_clients 8 --local_epochs 3 --output_dir ./outputs/fedprox_example
```

Train with **FedNova**:
```bash
python3 train.py --dataset cifar100 --agg fednova --rounds 40 \
    --num_clients 6 --local_epochs 2 --output_dir ./outputs/fednova_example
```

---

### 3. Run the full iNaturalist pipeline (GPU)

The script below automates the entire process:  
data download, verification, training, and evaluation.

```bash
bash full_pipeline_gpu.sh
```

It performs:
1. CUDA availability check  
2. Dataset preparation (iNaturalist 2021 mini)  
3. Verification of *Plantae* folders  
4. Federated + centralized training  
5. Evaluation and visualization  

Results and plots are stored in `./outputs/`.

---

### 4. Evaluate trained models

After training, evaluate the federated and centralized models:

```bash
python3 evaluate.py --experiment_dir ./outputs/fedavg_example --plot
```

This computes:
- Accuracy and loss  
- Per-class accuracy, precision, recall, and F1  
- Normalized confusion matrices (optional plots)

Generated outputs:
- `results.json` — metrics in structured format  
- `plots/` — per-class accuracy bars and confusion matrices  

---

##  Arguments Summary

| Argument           | Default                  | Description                                                        |
| ------------------ | ------------------------ | ------------------------------------------------------------------ |
| `--dataset`        | `cifar100`               | Dataset name (`cifar100` or `inaturalist`)                         |
| `--num_classes`    | `None`                   | Restrict number of classes (top-N for folder datasets)             |
| `--random_subset`  | `False`                  | Randomly sample subset of classes                                  |
| `--model`          | `mobilenet_v3_small`     | Backbone model architecture (`mobilenet_v3_small`, `resnet18`)     |
| `--agg`            | `fedavg`                 | Aggregation rule (`fedavg`, `fedprox`, `fednova`, `fedavgm`)       |
| `--mu`             | `0.0`                    | FedProx proximal coefficient                                       |
| `--rounds`         | `10`                     | Number of communication rounds                                     |
| `--local_epochs`   | `1`                      | Local training epochs per client                                   |
| `--central_epochs` | `5`                      | Epochs for centralized training baseline                           |
| `--num_clients`    | `4`                      | Number of participating clients                                    |
| `--batch_size`     | `32`                     | Training batch size                                                |
| `--alpha`          | `0.5`                    | Dirichlet alpha for non-IID splitting                              |
| `--lr`             | `1e-3`                   | Learning rate                                                      |
| `--seed`           | `42`                     | Random seed for reproducibility                                    |
| `--output_dir`     | `./outputs/experiment_1` | Output directory for logs, models, and metrics                     |

---

##  Project Structure

```
FedTree/
├── client.py             # Local training loop for each client (FedAvg/FedProx)
├── server.py             # Global model aggregation (FedAvg, FedNova, FedAvgM)
├── train.py              # Main training script (federated + centralized)
├── evaluate.py           # Evaluation and visualization of results
├── datasets.py           # Dataset loader (CIFAR-100, iNaturalist)
├── models.py             # Model definitions (MobileNetV3, ResNet18)
├── full_pipeline_gpu.sh  # Automated pipeline for iNaturalist (GPU setup)
└── outputs/              # Directory for trained models and evaluation results
```

---

##  Example Output

Example terminal output after evaluation:

```
Federated   -> Loss: 0.8421 | Accuracy: 72.35%
Centralized -> Loss: 0.7992 | Accuracy: 74.80%

Per-Class Accuracy (Fed vs Central):
maple_tree           | Fed: 70.00% | Central: 72.50%
oak_tree             | Fed: 68.00% | Central: 74.00%
palm_tree            | Fed: 75.00% | Central: 73.00%
...
```

---

##  Notes

- The **Dirichlet parameter** (`--alpha`) controls client heterogeneity:
  - Smaller `alpha` → more non-IID (clients have different class distributions)
- Folder-based datasets (e.g., iNaturalist) filter subfolders containing `"plantae"`.
- Compatible with both **CPU** and **GPU**.
- Outputs include both **federated** and **centralized** models for fair comparison.

