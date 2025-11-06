import torch
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

class Server:
    def __init__(self, global_model, aggregation='fedavg', device='cpu', momentum=0.9):
        """
        Federated server that maintains and updates the global model.

        Args:
            global_model (nn.Module): Initial global model shared with clients.
            aggregation (str): Aggregation rule, e.g. 'fedavg', 'fedprox', 'fednova', 'fedavgm'.
            device (torch.device): Device to store and update the global model.
            momentum (float): Used only if aggregation == 'fedavgm' (server momentum).
        """
        self.global_model = deepcopy(global_model).to(device)
        self.aggregation = aggregation.lower()
        self.device = device
        self.round = 0  # Track communication rounds

        # FedAvgM parameters (used for momentum-based aggregation)
        self.momentum_buffer = None
        self.momentum = momentum

    def get_model(self):
        """Return a deepcopy of the current global model for client training."""
        return deepcopy(self.global_model)

    def aggregate(self, client_states: List[Dict[str, torch.Tensor]],
              client_sizes: List[int],
              client_steps: List[int] = None):
        """
        Aggregate local client models into the global model.

        Args:
            client_states (List[Dict]): List of state_dicts from client models.
            client_sizes (List[int]): Number of training samples per client (used for weighting).
            client_steps (List[int], optional): Number of local steps per client (FedNova only).

        Raises:
            ValueError: If FedNova aggregation is selected but `client_steps` is not provided.
            NotImplementedError: If the specified aggregation method is unsupported.
        """
        total_size = sum(client_sizes)
        new_state = {}

        # -----------------------------------------------------------
        # FedAvg: Weighted average of client parameters
        # -----------------------------------------------------------
        if self.aggregation == 'fedavg':
            for key in client_states[0]:
                new_state[key] = sum(
                    state[key].float() * (size / total_size)
                    for state, size in zip(client_states, client_sizes)
                )

        # -----------------------------------------------------------
        # FedProx: Same aggregation rule as FedAvg (prox term handled client-side)
        # -----------------------------------------------------------
        elif self.aggregation == 'fedprox':
            for key in client_states[0]:
                new_state[key] = sum(
                    state[key].float() * (size / total_size)
                    for state, size in zip(client_states, client_sizes)
                )

        # -----------------------------------------------------------
        # FedNova: Normalize updates by local steps to reduce client drift
        # -----------------------------------------------------------
        elif self.aggregation == 'fednova':
            if client_steps is None:
                raise ValueError("FedNova requires `client_steps` input.")
            total_steps = sum(client_steps)
            baseline = self.global_model.state_dict()
            new_state = {key: torch.zeros_like(val) for key, val in baseline.items()}

            # Compute normalized weighted updates per client
            for c_state, c_step in zip(client_states, client_steps):
                for key in c_state:
                    if not baseline[key].is_floating_point():
                        continue  # Skip non-trainable parameters (e.g., buffers)
                    update = c_state[key] - baseline[key]
                    new_state[key] += (c_step / total_steps) * update

            # Apply aggregated updates to baseline
            for key in new_state:
                if not baseline[key].is_floating_point():
                    new_state[key] = baseline[key]  # Keep non-float params unchanged
                else:
                    new_state[key] = baseline[key] + new_state[key]

        # -----------------------------------------------------------
        # Unsupported aggregation method
        # -----------------------------------------------------------
        else:
            raise NotImplementedError(f"Aggregation '{self.aggregation}' is not supported.")

        # Update global model with aggregated parameters
        self.global_model.load_state_dict({k: v.to(self.device) for k, v in new_state.items()})
        self.round += 1  # Increment communication round counter
