from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

import torch
from torch.utils.data import DataLoader

from dual_agent.config import load_config
from dual_agent.models.surrogate import OPFSurrogate
from dual_agent.training.dataset import TeacherDataset, make_complete_graph
from dual_agent.training.loops import train_surrogate_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ieee13.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="surrogate.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = TeacherDataset(args.data)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes = dataset.pv_scenarios.size(-1)
    adjacency = make_complete_graph(nodes)

    model = OPFSurrogate(
        pv_nodes=nodes,
        load_nodes=nodes,
        horizon_steps=config.problem.horizon_steps,
        scenarios=config.problem.scenarios,
        load_feature_dim=config.problem.load_feature_dim,
        hidden_dim=config.model.hidden_dim,
        graph_layers=config.model.graph_layers,
        temporal_layers=config.model.temporal_layers,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    for epoch in range(config.training.epochs):
        loss = train_surrogate_epoch(model, loader, optimizer, adjacency, device)
        print(f"epoch={epoch + 1:03d} surrogate_loss={loss:.6f}")

    torch.save(model.state_dict(), args.checkpoint)


if __name__ == "__main__":
    main()
