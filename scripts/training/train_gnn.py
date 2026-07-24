# scripts/training/train_gnn.py

# -----------
# IMPORTS
# -----------
import time
import argparse

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader

from rsd.data import get_split_indices
from rsd.graphs import GraphDataset
from rsd.models import GraphModel, GraphNetwork
from rsd.utils import PATHS, MockID, load_config

# -----------
# MAIN
# -----------

parser = argparse.ArgumentParser(description='train GNN on galaxy graphs')
parser.add_argument('-n', '--n-graphs', type=int,
                    help='override the config subsample (0 = all)')
parser.add_argument('--smoke', action='store_true',
                    help='tiny run (512 graphs, 2 epochs)')
args = parser.parse_args()

cfg = load_config('model')
if args.n_graphs is not None:
    cfg['n_graphs'], cfg['frac_graphs'] = args.n_graphs, 0.  # explicit count wins over fraction
if args.smoke:
    cfg['n_graphs'], cfg['frac_graphs'] = min(cfg['n_graphs'], 512), 0.
    cfg['epochs'] = min(cfg['epochs'], 2)

if cfg["epochs"] < 1:
    raise ValueError(f"epochs must be positive, got {cfg['epochs']}")
if cfg["batch_size"] < 1:
    raise ValueError(f"batch_size must be positive, got {cfg['batch_size']}")
if cfg.get("n_graphs", 0) < 0:
    raise ValueError(f"n_graphs cannot be negative, got {cfg['n_graphs']}")
if not 0.0 <= cfg.get("frac_graphs", 0.0) <= 1.0:
    raise ValueError(f"frac_graphs must be in [0, 1], got {cfg['frac_graphs']}")
if cfg.get("lambda_pair", 1.0) < 0.0:
    raise ValueError(f"lambda_pair cannot be negative, got {cfg['lambda_pair']}")

seed = cfg.get("seed", 42)
torch.set_num_threads(cfg.get("threads", 1))
L.seed_everything(seed, workers=True)
rng = np.random.default_rng(seed)

# ---- load data ----
mock = MockID.from_cfg(load_config())
graph_path = PATHS.graph_pt(mock)

start = time.perf_counter()
dataset = GraphDataset(graph_path)

train_idx, val_idx, test_idx = get_split_indices(
    length=len(dataset),
    test_size=cfg.get("test_size", 0.1),
    val_size=cfg.get("val_size", 0.05),
    random_state=seed,
)

train_dataset = dataset.index_select(train_idx)
val_dataset = dataset.index_select(val_idx)
test_dataset = dataset.index_select(test_idx)

# ---- dataloaders ----
num_workers = cfg.get("workers", 0)

loader_options = {
    "batch_size": cfg["batch_size"],
    "num_workers": num_workers,
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": num_workers > 0,
}

train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

# ---- model ----
network = GraphNetwork(
    node_dim=dataset[0].num_node_features + 1,
    edge_dim=5,
    hidden=cfg["hidden"],
    n_layers=cfg["layers"],
    max_step=cfg.get("max_step", 0.5),
)
model = GraphModel(network, cfg)

run_name = f"fog_"

n_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
print(f"{run_name}: {n_parameters:,} trainable parameters")

# ---- logging and checking ----

logger = WandbLogger(
        project="rsd",
        name=run_name,
        save_dir=PATHS.results / "wandb" / "gnn",
        log_model=True,
    )

checkpoint = ModelCheckpoint(
    dirpath=PATHS.results / "ckpt" / "gnn",
    filename=run_name + "_e{epoch:02d}",
    monitor="val/loss",
    mode="min",
    save_top_k=1,
    auto_insert_metric_name=False,
)

trainer = L.Trainer(
    max_epochs=cfg["epochs"],
    accelerator="auto",
    devices=1,
    logger=logger,
    callbacks=[checkpoint],
    log_every_n_steps=10,
    default_root_dir=PATHS.results / "lightning" / "gnn",
)

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

print(f"Best checkpoint: {checkpoint.best_model_path} (val/loss={checkpoint.best_model_score})")

trainer.test(
    model,
    dataloaders=test_loader,
    ckpt_path="best",
)