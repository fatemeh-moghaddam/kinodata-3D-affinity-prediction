"""
The DTI probing readout: which artifacts it emits, and how it pools them.

These pin the two properties the probing comparison against CGNN/CGNN-3D rests on:

  1. `repr_aggr` swaps sum pooling for a normalized softmax mean in the *readout
     only*. The prediction path keeps the decoder's trained sum pooling, so a
     "DTI-soft" run must reproduce plain DTI's predictions exactly.
  2. Sum pooling leaks graph size into the representation and softmax pooling does
     not -- which is the asymmetry against `ComplexTransformer`'s SoftmaxAggregation
     that the variant exists to measure.
"""
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import HeteroData

import kinodata.configuration as cfg
from kinodata.data.featurization.atoms import AtomFeatures
from kinodata.data.featurization.bonds import NUM_BOND_TYPES
from kinodata.data.featurization.residue import known_residues
from kinodata.model.dti import _softmax_aggr, _sum_aggr, make_model
from kinodata.types import NodeType, RelationType

HIDDEN = 32
NUM_LIGAND_LAYERS = 4
NUM_POCKET_BLOCKS = 2
NUM_RESIDUES = 85

BASE_CONFIG = dict(
    loss_type="mse",
    hidden_channels=HIDDEN,
    num_layers=NUM_LIGAND_LAYERS,
    act="silu",
    num_attention_blocks=NUM_POCKET_BLOCKS,
    num_heads=1,
    residue_featurization="onehot",
    out_channels=1,
    optim="adamw",
    lr=1e-3,
)


def _build(**overrides):
    return make_model(cfg.Config({**BASE_CONFIG, **overrides})).eval()


def _batch(atoms_per_graph=(7, 7, 7), seed=0):
    """A minimal two-tower batch: ligand atoms + bonds, and 85 pocket residues each."""
    gen = torch.Generator().manual_seed(seed)
    n_graphs = len(atoms_per_graph)
    n_atoms = sum(atoms_per_graph)
    data = HeteroData()

    ligand = data[NodeType.Ligand]
    ligand.z = torch.randint(1, 20, (n_atoms,), generator=gen)
    ligand.x = torch.randn(n_atoms, AtomFeatures.size, generator=gen)
    ligand.batch = torch.arange(n_graphs).repeat_interleave(
        torch.tensor(atoms_per_graph)
    )

    bonds = data[NodeType.Ligand, RelationType.Covalent, NodeType.Ligand]
    bonds.edge_index = torch.randint(0, n_atoms, (2, 5 * n_atoms), generator=gen)
    bonds.edge_attr = torch.randn(5 * n_atoms, NUM_BOND_TYPES, generator=gen)

    pocket = data[NodeType.PocketResidue]
    pocket.x = torch.randn(
        n_graphs * NUM_RESIDUES, len(known_residues) + 1, generator=gen
    )
    pocket.batch = torch.arange(n_graphs).repeat_interleave(NUM_RESIDUES)
    return data


JOINT_LAYERS = {f"layer_{i}" for i in range(NUM_LIGAND_LAYERS + 1)}
TOWER_LAYERS = {f"ligand_layer_{i}" for i in range(NUM_LIGAND_LAYERS + 1)} | {
    f"pocket_layer_{i}" for i in range(NUM_POCKET_BLOCKS + 1)
}


def test_tower_representations_are_off_by_default():
    """A default run leaves only the joint layers the probing actually reads."""
    with torch.no_grad():
        _, reprs, _, _ = _build()(_batch())
    assert set(reprs) == JOINT_LAYERS


def test_emit_tower_reprs_adds_towers_without_touching_joint_layers():
    off, on = _build(), _build(emit_tower_reprs=True)
    on.load_state_dict(off.state_dict())
    batch = _batch()
    with torch.no_grad():
        _, reprs_off, _, _ = off(batch)
        _, reprs_on, _, _ = on(batch)

    assert set(reprs_on) == JOINT_LAYERS | TOWER_LAYERS
    for name in reprs_off:
        assert torch.equal(reprs_off[name][0], reprs_on[name][0])


def test_probe_layer_names_matches_what_the_model_emits():
    """
    A name listed but never emitted makes the resume check wait on a file that will
    never be written, so these two must not drift apart.
    """
    run_extraction = pytest.importorskip("prob.run_extraction")
    batch = _batch()
    for emit in (False, True):
        with torch.no_grad():
            _, reprs, _, _ = _build(emit_tower_reprs=emit)(batch)
        config = cfg.Config({**BASE_CONFIG, "emit_tower_reprs": emit})
        for model_type in ("DTI", "DTI-soft"):
            assert set(run_extraction.probe_layer_names(config, model_type)) == set(reprs)


def test_softmax_readout_reuses_the_checkpoint_and_preserves_predictions():
    """
    DTI-soft is the *same weights* read out differently: the state dict has to load
    strictly, and only layer_0..N-1 may move. layer_N is the decoder's combined
    vector, which keeps its trained sum pooling.
    """
    summed = _build()
    softmax = _build(repr_aggr="softmax")
    softmax.load_state_dict(summed.state_dict())  # strict: no new parameters

    batch = _batch()
    with torch.no_grad():
        pred_sum, reprs_sum, _, combined_sum = summed(batch)
        pred_soft, reprs_soft, _, combined_soft = softmax(batch)

    for depth in range(NUM_LIGAND_LAYERS):
        name = f"layer_{depth}"
        assert not torch.allclose(reprs_sum[name][0], reprs_soft[name][0])

    last = f"layer_{NUM_LIGAND_LAYERS}"
    assert torch.equal(reprs_sum[last][0], reprs_soft[last][0])
    assert torch.equal(combined_sum, combined_soft)
    assert torch.equal(pred_sum, pred_soft)


def test_sum_leaks_graph_size_and_softmax_does_not():
    """
    Graph 1 holds graph 0's atom set repeated twice -- same composition, twice the
    count. Compared before any message passing (layer_0's ligand half), sum doubles
    and softmax, being a weighted mean, does not move.
    """
    batch = _batch(atoms_per_graph=(6, 12), seed=3)
    ligand = batch[NodeType.Ligand]
    z, x = ligand.z[:6].clone(), ligand.x[:6].clone()
    ligand.z = torch.cat([z, z, z])
    ligand.x = torch.cat([x, x, x])

    summed = _build()
    softmax = _build(repr_aggr="softmax")
    softmax.load_state_dict(summed.state_dict())
    with torch.no_grad():
        pooled_sum = summed(batch)[1]["layer_0"][0][:, :HIDDEN]
        pooled_soft = softmax(batch)[1]["layer_0"][0][:, :HIDDEN]

    assert torch.allclose(pooled_sum[1], 2 * pooled_sum[0], atol=1e-5)
    assert torch.allclose(pooled_soft[1], pooled_soft[0], atol=1e-5)


def test_softmax_aggr_matches_pyg_reference():
    """`_softmax_aggr` must be SoftmaxAggregation(learn=False), on both code paths."""
    from torch_geometric.nn.aggr import SoftmaxAggregation

    reference = SoftmaxAggregation(learn=False, t=1.0)
    gen = torch.Generator().manual_seed(0)

    sparse = torch.randn(11, 5, generator=gen)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    assert torch.allclose(_softmax_aggr(sparse, index), reference(sparse, index))

    dense = torch.randn(4, NUM_RESIDUES, 5, generator=gen)
    per_graph = torch.zeros(NUM_RESIDUES, dtype=torch.long)
    assert torch.allclose(
        _softmax_aggr(dense, None)[0], reference(dense[0], per_graph)[0], atol=1e-6
    )


def test_sum_aggr_is_unchanged():
    """The default path must still be exactly what the trained model used."""
    gen = torch.Generator().manual_seed(1)
    sparse = torch.randn(9, 4, generator=gen)
    index = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2])
    expected = torch.stack([sparse[:4].sum(0), sparse[4:6].sum(0), sparse[6:].sum(0)])
    assert torch.allclose(_sum_aggr(sparse, index), expected)

    dense = torch.randn(3, NUM_RESIDUES, 4, generator=gen)
    assert torch.allclose(_sum_aggr(dense, None), dense.sum(dim=1))


def test_rejects_unknown_repr_aggr():
    with pytest.raises(ValueError, match="repr_aggr"):
        _build(repr_aggr="nonsense")
