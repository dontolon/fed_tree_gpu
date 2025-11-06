#!/usr/bin/env python3

import argparse
import os
import json
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

from datasets import TreeSubsetDataset, get_transform
from models import get_model
from client import train_local
from server import Server


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Ensure reproducibility across runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_clients(class_indices, num_clients, alpha):
    """
    Split dataset among clients using a Dirichlet distribution.

    Args:
        class_indices (dict): Mapping of class -> sample indices.
        num_clients (int): Number of federated clients.
        alpha (float): Dirichlet parameter controlling data heterogeneity.
                       Lower alpha = more non-IID distribution.

    Returns:
        list: A list of client index lists.
    """
    client_idxs = [[] for _ in range(num_clients)]
    for c, idxs in class_indices.items():
        idxs = np.array(idxs)
        np.random.shuffle(idxs)
        props = np.random.dirichlet(np.repeat(alpha, num_clients))
        cuts = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, cuts)
        for i in range(num_clients):
            client_idxs[i].extend(splits[i].tolist())
    return client_idxs


def train_centralized(model, dataloader, device, lr, epochs):
    """
    Train a single centralized model on the full dataset (baseline).

    Args:
        model: Neural network model to train.
        dataloader: DataLoader for the entire training dataset.
        device: torch device (cpu or cuda).
        lr (float): Learning rate.
        epochs (int): Number of epochs to train.

    Returns:
        model: Trained model (moved to CPU).
        losses: List of epoch-level training losses.
    """
    print("Training centralized model...")
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []

    for ep in range(epochs):
        total_loss = 0
        total_samples = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
        epoch_loss = total_loss / total_samples
        losses.append(epoch_loss)
        print(f"Centralized Epoch {ep+1}/{epochs} | Loss: {epoch_loss:.4f}")

    return model.cpu(), losses


# ---------------------------------------------------------------------------
# Main federated learning procedure
# ---------------------------------------------------------------------------

def main(args):
    """Main entry point for running the federated training and baseline."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Dataset loading and preprocessing
    # ---------------------------------------------------------
    transform = get_transform(model_type=args.model)
    train_ds = TreeSubsetDataset(
        args.dataset, root="./data", train=True,
        transform=transform,
        num_classes=args.num_classes,
        random_subset=args.random_subset
    )
    test_ds = TreeSubsetDataset(
        args.dataset, root="./data", train=False,
        transform=transform,
        num_classes=args.num_classes,
        class_names=train_ds.class_names
    )

    num_classes = len(train_ds.class_names)
    print(f"Loaded dataset: {args.dataset} | Classes: {num_classes}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(test_ds)}")

    # ---------------------------------------------------------
    # Client creation and data partitioning
    # ---------------------------------------------------------
    class_indices = train_ds.get_class_indices()
    client_idxs = split_clients(class_indices, args.num_clients, args.alpha)
    client_loaders = [
        DataLoader(Subset(train_ds, idxs), batch_size=args.batch_size, shuffle=True)
        for idxs in client_idxs
    ]

    # ---------------------------------------------------------
    # Initialize server and global model
    # ---------------------------------------------------------
    global_model = get_model(args.model, num_classes=num_classes, pretrained=True)
    server = Server(global_model, aggregation=args.agg, device=device)

    # Initialize tracking lists
    fed_avg_losses = []
    val_losses = []
    fed_client_losses = [[] for _ in range(args.num_clients)]

    print("\nStarting federated training...")
    criterion = torch.nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Federated training rounds
    # ---------------------------------------------------------
    for rnd in range(1, args.rounds + 1):
        client_states = []
        client_sizes = []
        round_losses = []

        # Use global model snapshot for FedProx
        global_state = None
        if args.agg == 'fedprox':
            global_state = server.get_model().state_dict()

        # --- Local training on each client ---
        for cid, loader in enumerate(client_loaders):
            local_model = server.get_model()
            state_dict, num_samples, loss = train_local(
                local_model, loader, device,
                lr=args.lr, epochs=args.local_epochs,
                global_state=global_state, mu=args.mu if args.agg == 'fedprox' else 0.0
            )
            client_states.append(state_dict)
            client_sizes.append(num_samples)
            fed_client_losses[cid].append(loss)
            round_losses.append(loss)

        # --- Aggregate client models ---
        avg_loss = sum(sz * l for sz, l in zip(client_sizes, round_losses)) / sum(client_sizes)
        fed_avg_losses.append(avg_loss)
        print(f"\nRound {rnd}/{args.rounds} | Train loss: {avg_loss:.4f}")

        if args.agg == 'fednova':
            server.aggregate(client_states, client_sizes, client_steps=[args.local_epochs]*len(client_states))
        else:
            server.aggregate(client_states, client_sizes)

        # --- Validation phase (using test_ds as validation) ---
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        server_model = server.get_model().to(device)
        server_model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = server_model(imgs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)
        val_loss = total_loss / total_samples
        val_losses.append(val_loss)
        print(f"Validation loss after round {rnd}: {val_loss:.4f}")

    # ---------------------------------------------------------
    # Save federated training artifacts
    # ---------------------------------------------------------
    np.save(os.path.join(args.output_dir, "fed_losses.npy"), np.array(fed_avg_losses))
    np.save(os.path.join(args.output_dir, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(args.output_dir, "per_client_losses.npy"), np.array(fed_client_losses, dtype=object))
    torch.save(server.get_model().state_dict(), os.path.join(args.output_dir, "fed_model.pt"))

    print("\nFederated training complete.")
    print(f"Final Train Loss: {fed_avg_losses[-1]:.4f} | Final Val Loss: {val_losses[-1]:.4f}")

    # ---------------------------------------------------------
    # Centralized baseline training for comparison
    # ---------------------------------------------------------
    centralized_model = get_model(args.model, num_classes=num_classes, pretrained=True)
    centralized_model, central_losses = train_centralized(
        centralized_model,
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        device,
        lr=args.lr,
        epochs=args.central_epochs
    )

    np.save(os.path.join(args.output_dir, "central_losses.npy"), np.array(central_losses))
    torch.save(centralized_model.state_dict(), os.path.join(args.output_dir, "central_model.pt"))

    # ---------------------------------------------------------
    # Save metadata for evaluation scripts
    # ---------------------------------------------------------
    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "aggregation": args.agg,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "central_epochs": args.central_epochs,
        "lr": args.lr,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "mu": args.mu,
        "seed": args.seed,
        "classes": train_ds.class_names
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nAll training artifacts saved to:", args.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--random_subset', action='store_true')
    parser.add_argument('--model', default='mobilenet_v3_small')
    parser.add_argument('--agg', default='fedavg', choices=['fedavg', 'fedavgm', 'fedprox', 'fednova'])
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--central_epochs', type=int, default=5)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='./outputs/experiment_1')
    args = parser.parse_args()
    main(args)
