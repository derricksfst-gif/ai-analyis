"""Kokumi / non-kokumi inference with the published WeaveGNN weights."""

import os

os.environ.setdefault("DGLBACKEND", "pytorch")

from pathlib import Path

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import WeaveGNN
from dgllife.model.readout import WeightedSumAndMax
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from rdkit import Chem

WEIGHT_PATH = Path(__file__).resolve().parent / "新功能" / "kn_gnn.pth"
LABELS = ("Kokumi", "Non-Kokumi")


class WeavePredictor(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        num_gnn_layers=2,
        gnn_hidden_feats=128,
        gnn_activation=F.relu,
        n_tasks=1,
    ):
        super().__init__()

        self.gnn = WeaveGNN(
            node_in_feats=node_in_feats,
            edge_in_feats=edge_in_feats,
            num_layers=num_gnn_layers,
            hidden_feats=gnn_hidden_feats,
            activation=gnn_activation,
        )
        self.readout = WeightedSumAndMax(gnn_hidden_feats)
        self.predict = nn.Sequential(
            nn.Linear(2 * gnn_hidden_feats, 64),
            nn.Linear(64, n_tasks),
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats, node_only=True)
        g_feats = self.readout(g, node_feats)
        return self.predict(g_feats)


def graph_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    if molecule.GetNumBonds() == 0:
        return None
    return mol_to_bigraph(
        molecule,
        node_featurizer=CanonicalAtomFeaturizer(atom_data_field="h"),
        edge_featurizer=CanonicalBondFeaturizer(bond_data_field="e"),
    )


def collate_smiles(graphs):
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph


def load_model(weight_path=WEIGHT_PATH):
    model = WeavePredictor(node_in_feats=74, edge_in_feats=12, n_tasks=2)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_smiles(model, smiles):
    """Return (kokumi probability, label). Invalid molecules return (None, 'Invalid')."""
    if smiles is None or (isinstance(smiles, float) and smiles != smiles):
        return None, "Invalid"
    text = str(smiles).strip()
    if not text or text.lower() == "nan":
        return None, "Invalid"

    graph = graph_from_smiles(text)
    if graph is None:
        return None, "Invalid"

    batched = collate_smiles([graph])
    with torch.no_grad():
        logits = model(batched, batched.ndata.pop("h"), batched.edata.pop("e"))
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    kokumi_probability = float(probabilities[0])
    label = LABELS[0] if kokumi_probability >= float(probabilities[1]) else LABELS[1]
    return kokumi_probability, label
