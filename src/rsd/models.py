# src/models.py
# ------------------
# Imports
# ------------------

from collections.abc import Sequence
import io
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, scatter
import wandb

# ------------------
# Functions
# ------------------

def log_matplotlib_figure(fig: plt.Figure, figure_label: str):
    """log a matplotlib figure to wandb, avoiding plotly

    Args:
        figure_label (str): label for figure
    """
    # Save plot to a buffer, otherwise wandb does ugly plotly
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    image = Image.open(buf)
    # Log the plot to wandb
    wandb.log({f"{figure_label}": wandb.Image(image)})


def _fig_pred_vs_true(delta, y):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, delta, s=4, alpha=0.3, edgecolors='none')
    lo, hi = min(y.min(), delta.min()), max(y.max(), delta.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel(r'true $\delta$')
    ax.set_ylabel(r'pred $\delta$')
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    return fig


def _fig_pair_residual(res):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(res, bins=60)
    ax.axvline(0., color='k', ls='--', lw=1)
    ax.set_xlabel(r'pair residual $\delta_i-\delta_j+z_{ij}$')
    ax.set_ylabel('count')
    fig.tight_layout()
    return fig


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    *,
    use_layer_norm: bool = True,
    dropout_rate: float = 0.0,
) -> nn.Sequential:
    """Creates a multi-layer perceptron (MLP) with optional batch normalization and dropout."""
    dims = (input_dim, *tuple(hidden_dims), output_dim)
    if any(dim < 1 for dim in dims):
        raise ValueError(f"all dimensions must be positive, got {dims}")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout_rate}")

    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SiLU())
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


def update_edge_geom(data, delta):
    """updates the current edge features from predicted node corrections"""
    # update the edges w delta
    src, dst = data.edge_index
    
    z_current = data.s_par + delta
    delta_z = z_current[src] - z_current[dst]
    
    edge_rperp_sq = data.edge_rperp_sq.clamp_min(0.0)
    edge_rperp = edge_rperp_sq.sqrt()

    edge_distance = torch.sqrt(
        edge_rperp_sq
        + delta_z.square()
        + 1e-12
    )

    edge_to_center = (data.center_node[src] | data.center_node[dst]).float()
    
    # pass back the feature stack
    message_features = torch.stack([
        edge_rperp,
        data.edge_angle,
        edge_to_center,
        delta_z,
        edge_distance,
    ], dim=1)

    # edge gate should be symmetric under i <-> j
    gate_features = torch.stack([
        edge_rperp,
        data.edge_angle,
        edge_to_center,
        delta_z.abs(),
        edge_distance,
    ], dim=1)

    return message_features, gate_features, edge_distance


def batch_loss(model, batch, cfg):
    delta, logvar, gate = model(batch)
    
    center_node = batch.center_node.bool()
    target_delta = batch.delta_true
    variance = logvar.exp().clamp_min(1e-6)
    
    # nll for node positions
    node_nll = F.gaussian_nll_loss(
        delta[~center_node],
        target_delta[~center_node],
        variance[~center_node]
    )
    
    # pairwise errors
    src, dst = batch.edge_index
    unique = src < dst
    src, dst = src[unique], dst[unique]

    pair_error = (
        delta[src] - delta[dst]
        - (target_delta[src] - target_delta[dst])
    )
    pair_gate = gate[unique].clamp_min(1e-6)
    pair_graph = batch.batch[src]
    
    mean_gate = scatter(
        pair_gate, 
        pair_graph, 
        dim_size=batch.num_graphs, 
        reduce="mean"
    )
    
    pair_weight = pair_gate / (
        mean_gate[pair_graph].clamp_min(1e-6)
    )
    pair_mse = (
        pair_weight * pair_error.square()
    ).mean()
    
    loss = (
        node_nll
        + cfg['lambda_pair'] * pair_mse
    )
    
    metrics = {
        "loss": loss.detach(),
        "node_nll": node_nll.detach(),
        "pair_mse": pair_mse.detach(),
        "node_rmse": F.mse_loss(
            delta[~center_node],
            target_delta[~center_node],
        ).sqrt().detach(),
        "pair_rmse": (
            pair_error.square().mean().sqrt().detach()
        ),
        "sigma": (
            0.5 * logvar[~center_node]
        ).exp().mean().detach(),
        "gate_mean": pair_gate.mean().detach(),
    }
    
    return loss, metrics, (delta, logvar, gate)


@torch.no_grad()
def diagnostic_metrics(batch, delta):
    target_delta = batch.delta_true

    src, dst = batch.edge_index
    unique = src < dst
    src, dst = src[unique], dst[unique]

    pair_error = (
        delta[src] - delta[dst]
        - (target_delta[src] - target_delta[dst])
    )

    metrics = {}

    # graph center
    center_pair = batch.center_node[src]| batch.center_node[dst]
    if center_pair.any():
        metrics["pair_rmse_center"] = pair_error[center_pair].square().mean().sqrt()
    if (~center_pair).any():
        metrics["pair_rmse_noncenter"] = pair_error[~center_pair].square().mean().sqrt()
        
    # halo cent/sats
    halo_central = batch.is_central.bool()
    central_satellite = halo_central[src] ^ halo_central[dst]
    satellite_satellite = ~halo_central[src]& ~halo_central[dst]
    
    if central_satellite.any():
        metrics["pair_rmse_halo_cs"] = pair_error[central_satellite].square().mean().sqrt()
    if satellite_satellite.any():
        metrics["pair_rmse_halo_ss"] = pair_error[satellite_satellite].square().mean().sqrt()
    return metrics

# ------------------
# Classes
# ------------------

class GatedConv(MessagePassing):
    """Attention+gate convolution based message passing"""
    def __init__(self, hidden):
        super().__init__(aggr="add")
        self.message_mlp = build_mlp(3 * hidden, hidden, [hidden])
        self.attention_mlp = build_mlp(3 * hidden, 1, [hidden])
        self.update_mlp = build_mlp(2 * hidden, hidden, [hidden])
        # normalization residual node update
        self.norm = nn.LayerNorm(hidden)
        
    def forward(self,node_hidden,edge_index,edge_hidden,edge_weight):
        aggregated = self.propagate(
            edge_index,
            h=node_hidden,
            edge_hidden=edge_hidden,
            edge_weight=edge_weight,
        )
        update = self.update_mlp(
            torch.cat([node_hidden,aggregated], dim=1)
        )
        return self.norm(node_hidden + update)

    def message(self, h_i, h_j, edge_hidden, edge_weight, index, size_i):
        message_input = torch.cat([h_i, h_j, edge_hidden], dim=1)
        attention_logits = self.attention_mlp(message_input).squeeze(-1)
        attention = softmax(
            attention_logits,
            index,
            num_nodes=size_i,
        )
        learned_message = self.message_mlp(message_input)
        return (learned_message * attention.unsqueeze(1) * edge_weight.unsqueeze(1))


class GraphNetwork(nn.Module):
    def __init__(
        self, 
        node_dim=3, 
        edge_dim=5,
        hidden=64, 
        n_layers=3,
        max_step=0.5,
    ):
        super().__init__()
        if hidden < 1 or n_layers < 1:
            raise ValueError(
                f"hidden and n_layers must be positive, "
                f"got {hidden}, {n_layers}"
            )
        self.max_step = max_step
        
        self.node_encoder = build_mlp(
            input_dim=node_dim, 
            output_dim=hidden, 
            hidden_dims=[hidden]
        )
        self.edge_encoder = build_mlp(
            input_dim=edge_dim, 
            output_dim=hidden, 
            hidden_dims=[hidden]
        )
        self.edge_gate = build_mlp(
            input_dim=edge_dim, 
            output_dim=1, 
            hidden_dims=[32]
        )
        nn.init.constant_(
            self.edge_gate[-1].bias,
            2.0
        )
        
        self.layers = nn.ModuleList([
            GatedConv(hidden) 
            for _ in range(n_layers)
        ])
        
        # layers need to pred coordinate updates
        self.position_heads = nn.ModuleList([
            build_mlp(
                input_dim=hidden,
                output_dim=1,
                hidden_dims=[hidden],
            )
            for _ in range(n_layers)
        ])
        
        # gives uncertainty of delta
        self.logvar_head = build_mlp(
            input_dim=hidden,
            output_dim=1,
            hidden_dims=[hidden],
        )

    def forward(self, data):
        center_node = data.center_node.bool()
        
        node_in = torch.cat([
            data.x,
            center_node.float().unsqueeze(1),
        ], dim=1)
        node_hidden = self.node_encoder(node_in)
        
        # correction wrt s_par
        delta = torch.zeros_like(data.s_par)
        gate = None
        
        for layer, position_head in zip(self.layers, self.position_heads):
            message_features, gate_features, _ = update_edge_geom(data, delta)
            
            edge_hidden = self.edge_encoder(message_features)
            gate = torch.sigmoid(self.edge_gate(gate_features)).squeeze(-1)
            
            node_hidden = layer(
                node_hidden=node_hidden,
                edge_index=data.edge_index,
                edge_hidden=edge_hidden,
                edge_weight=gate,
            )
            
            delta_step = self.max_step * torch.tanh(position_head(node_hidden).squeeze(-1))
            # fixes center node in place while allowing node rep to update
            delta = delta + (~center_node).float() * delta_step
            delta = delta.masked_fill(center_node, 0.0)

        logvar = self.logvar_head(node_hidden).squeeze(-1)
        logvar = logvar.clamp(min=-10.0,max=10.0)
        
        return delta, logvar, gate

class GraphModel(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        if cfg['lr'] <= 0:
            raise ValueError(f"lr must be positive, got {cfg['lr']}")
        self.model = model
        self.cfg = cfg
        self._plot_batch = None
        self.save_hyperparameters(cfg)

    def training_step(self, batch, batch_idx):
        loss, metrics, _ = batch_loss(
            self.model,
            batch,
            self.cfg,
        )
        self.log_dict({f"train/{key}": value for key, value in metrics.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics, outputs = batch_loss(
            self.model,
            batch,
            self.cfg,
        )
        delta, _, _ = outputs

        metrics.update(diagnostic_metrics(
            batch=batch,
            delta=delta,
        ))
        self.log_dict({f"val/{key}": value for key, value in metrics.items()})
        if batch_idx == 0:  # stash a batch for epoch-end figs
            self._plot_batch = batch
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics, outputs = batch_loss(self.model, batch, self.cfg)
        delta, _, _ = outputs
        
        metrics.update(diagnostic_metrics(
            batch=batch,
            delta=delta,
        ))
        self.log_dict({f"test/{key}": value for key, value in metrics.items()})
        if batch_idx == 0:  # stash a batch for epoch-end figs
            self._plot_batch = batch
        return loss

    def on_validation_epoch_end(self):
        self._log_figures('val')

    def on_test_epoch_end(self):
        self._log_figures('test')

    def _log_figures(self, prefix):
        if not isinstance(self.logger, WandbLogger) or self._plot_batch is None:
            return
        figs = self._diag_figures(self._plot_batch, prefix)
        for label, fig in figs.items():
            log_matplotlib_figure(fig, label)
            plt.close(fig)
        self._plot_batch = None

    @torch.no_grad()
    def _diag_figures(self, batch, prefix):
        delta, _, _ = self.model(batch)

        center_node = batch.center_node.bool()
        target_delta = batch.delta_true

        src, dst = batch.edge_index
        unique = src < dst
        src, dst = src[unique], dst[unique]

        pair_error = (
            delta[src] - delta[dst]
            - (target_delta[src] - target_delta[dst])
        )

        return {
            f"{prefix}/pred_vs_true": _fig_pred_vs_true(
                delta[~center_node].cpu().numpy(),
                target_delta[~center_node].cpu().numpy(),
            ),
            f"{prefix}/pair_residual": _fig_pair_residual(
                pair_error.cpu().numpy()
            ),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg['lr'])

