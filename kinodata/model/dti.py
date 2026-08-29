from typing import Any, Callable, Optional, Protocol, Tuple

import torch
from kinodata.configuration import Config
from kinodata.data.featurization.bonds import NUM_BOND_TYPES
from kinodata.data.featurization.residue import known_residues
from kinodata.model.regression import RegressionModel
from kinodata.model.resolve import resolve_act, resolve_loss
from kinodata.model.shared import SetAttentionBlock
from kinodata.model.shared.gine import LigandGINE
from kinodata.types import NodeType
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, Module, ModuleList, Sequential, SiLU
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import softmax as segment_softmax
from torch_geometric.utils import to_dense_batch


class Encoder(Protocol):
    def __call__(self, batch: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        batch : Any
            pyg data object

        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Embeddings of size (N, d) (float)
            Index tensor of size (N,) (long), assigns embeddings to their batch.
        """
        ...


class Decoder(Protocol):
    def __call__(
        self,
        ligand_embeddings: Tensor,
        pocket_embeddings: Tensor,
        ligand_batch: Optional[Tensor] = None,
        pocket_batch: Optional[Tensor] = None,
    ) -> Tensor:
        ...


class DTIModel(RegressionModel):
    """
    Two-tower baseline: a ligand encoder and a pocket encoder whose pooled outputs
    the decoder combines into one prediction.

    `prob` mirrors `ComplexTransformer.prob`: when set, forward returns the same
    4-tuple the probing extraction expects instead of a bare prediction. Training
    needs a bare prediction, so leave it off there.

    `emit_tower_reprs` additionally writes the per-tower representations behind the
    joint ones. They are only needed for branch-resolved analysis and otherwise just
    triple the artifact count of a run, so they are off by default.

    `repr_aggr` selects how the probing readout pools node/residue states into one
    row per complex; see `_probe_pool`.
    """

    #: Pooling functions the probing readout can use, by `repr_aggr` name.
    REPR_AGGRS = ("sum", "softmax")

    def __init__(
        self,
        config: Config,
        ligand_encoder_cls: Callable[..., Encoder],
        pocket_encoder_cls: Callable[..., Encoder],
        decoder_cls: Callable[..., Decoder],
        prob: bool = True,
        emit_tower_reprs: bool = False,
        repr_aggr: str = "sum",
        repr_aggr_t: float = 1.0,
    ) -> None:
        super().__init__(config)
        if repr_aggr not in self.REPR_AGGRS:
            raise ValueError(
                f"Unknown repr_aggr {repr_aggr!r}, expected one of {self.REPR_AGGRS}"
            )
        self.criterion = resolve_loss(config.loss_type)
        self.ligand_encoder = config.init(ligand_encoder_cls)
        self.pocket_encoder = config.init(pocket_encoder_cls)
        self.decoder = config.init(decoder_cls)
        self.prob = prob
        self.emit_tower_reprs = emit_tower_reprs
        self.repr_aggr = repr_aggr
        self.repr_aggr_t = repr_aggr_t

    def _probe_pool(self, x: Tensor, index: Optional[Tensor] = None) -> Tensor:
        """
        Pool a tower's node/residue states into one row per complex, for probing only.

        The trained model sums, and that is what `repr_aggr="sum"` reproduces. Sum
        pooling keeps the magnitude of the total, so ||repr|| grows with the number
        of atoms/residues and the complex's size is legible straight off the vector.
        `ComplexTransformer` instead pools with `SoftmaxAggregation`, whose weights
        sum to one per graph per channel -- a weighted *mean*, from which size has
        been divided out. Probing results are not comparable across a difference like
        that, so `repr_aggr="softmax"` re-reads the same trained DTI weights through
        the normalizing pooling to separate the two effects.

        The softmax here is deliberately parameter-free (the `learn=False` form of
        `SoftmaxAggregation`, temperature `repr_aggr_t`): the checkpoint is frozen, so
        a learnable per-channel temperature would be randomly initialised and would
        break the strict `load_state_dict`. It is therefore not bit-identical to the
        temperature `ComplexTransformer` trained -- it isolates sum-vs-normalized-mean,
        not CGNN's exact operator.

        This affects the *readout only*. The prediction path keeps the decoder's
        trained sum pooling untouched, so predictions and the `layer_{N}` decoder
        representation are identical whichever `repr_aggr` is set; feeding softmax-
        pooled (roughly 1/N-scaled) inputs to `f_ligand`/`f_pocket` would take them
        far off the distribution they were trained on.
        """
        if self.repr_aggr == "softmax":
            return _softmax_aggr(x, index, t=self.repr_aggr_t)
        return _sum_aggr(x, index)

    def forward(self, batch):
        if self.prob:
            return self._forward_prob(batch)
        x_ligand, batch_ligand = self.ligand_encoder(batch)
        x_pocket, batch_pocket = self.pocket_encoder(batch)
        prediction = self.decoder(x_ligand, x_pocket, batch_ligand, batch_pocket)
        return prediction

    def _forward_prob(self, batch):
        """
        Forward pass that also reports per-layer *graph-level* representations.

        The two towers pool differently -- the ligand tower over a variable number
        of atoms via its batch index, the pocket tower over the fixed 85 residues of
        a dense tensor -- so unlike `ComplexTransformer` there is no single `aggr` a
        caller could apply afterwards. Pooling therefore happens here, via
        `_probe_pool` (see there for what `repr_aggr` changes), and the returned
        tensors are already one row per complex (signalled to the extractor by a
        `None` batch index).
        """
        if not hasattr(self.decoder, "combined_representation"):
            raise NotImplementedError(
                f"prob mode needs a decoder exposing `combined_representation`, "
                f"got {type(self.decoder).__name__}"
            )

        x_ligand, batch_ligand, ligand_layers = self.ligand_encoder(
            batch, return_intermediates=True
        )
        x_pocket, batch_pocket, pocket_layers = self.pocket_encoder(
            batch, return_intermediates=True
        )
        combined = self.decoder.combined_representation(
            x_ligand, x_pocket, batch_ligand, batch_pocket
        )
        prediction = self.decoder.f_combined(combined)

        ligand_pooled = [self._probe_pool(x, batch_ligand).detach() for x in ligand_layers]
        pocket_pooled = [self._probe_pool(x, batch_pocket).detach() for x in pocket_layers]

        graph_reprs = {}
        if self.emit_tower_reprs:
            for depth, x in enumerate(ligand_pooled):
                graph_reprs[f"ligand_layer_{depth}"] = (x, None)
            for depth, x in enumerate(pocket_pooled):
                graph_reprs[f"pocket_layer_{depth}"] = (x, None)

        # Depth-aligned joint representations, comparable to ComplexTransformer's
        # `layer_i`. The towers can differ in depth (4 GINE layers vs 2 attention
        # blocks in the trained baseline), so a tower that runs out keeps its last
        # layer. The deepest one is the decoder input -- the analogue of the pooled
        # representation ComplexTransformer feeds to its head.
        #
        # That last one is the odd one out in two ways, and cross-model comparisons
        # should generally stop one short of it. It is the only `layer_*` carrying
        # the decoder's own trained f_ligand/f_pocket MLPs, which ComplexTransformer
        # has no counterpart to (its pooled `graph_repr` is returned separately and
        # the extractor drops it); and it always comes from the decoder's trained sum
        # pooling, so `repr_aggr` does not reach it.
        last_pocket = len(pocket_pooled) - 1
        for depth in range(len(ligand_pooled) - 1):
            graph_reprs[f"layer_{depth}"] = (
                torch.cat(
                    (ligand_pooled[depth], pocket_pooled[min(depth, last_pocket)]),
                    dim=-1,
                ),
                None,
            )
        graph_reprs[f"layer_{len(ligand_pooled) - 1}"] = (combined.detach(), None)

        return prediction, graph_reprs, {}, combined.detach()


class ResidueTransformer(Module):
    def __init__(
        self,
        residue_size: int,
        hidden_channels: int,
        num_attention_blocks: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.lin = Sequential(
            Linear(residue_size, hidden_channels), SiLU(), LayerNorm(hidden_channels)
        )
        self.positional_encoding = Embedding(85, hidden_channels)
        self.attention_blocks = ModuleList(
            [
                SetAttentionBlock(hidden_channels, num_heads)
                for _ in range(num_attention_blocks)
            ]
        )

    def get_residue_representation(self, batch):
        x, _ = to_dense_batch(
            batch[NodeType.PocketResidue].x, batch[NodeType.PocketResidue].batch
        )
        return x

    def forward(self, batch, return_intermediates: bool = False):
        """
        With `return_intermediates`, additionally returns the residue representations
        after every attention block, starting with the projected input (before any
        attention). Used for probing; the default path is unchanged.
        """
        x = (
            self.lin(self.get_residue_representation(batch))
            + self.positional_encoding.weight
        )
        intermediates = [x]
        for attn in self.attention_blocks:
            x = attn(x)
            if return_intermediates:
                intermediates.append(x)
        if return_intermediates:
            return x, None, intermediates
        return x, None


class KissimTransformer(ResidueTransformer):
    def __init__(
        self,
        residue_size: int,
        hidden_channels: int,
        num_attention_blocks: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__(residue_size, hidden_channels, num_attention_blocks, num_heads)

    def get_residue_representation(self, batch):
        return batch.kissim_fp.float()


def _sum_aggr(
    x: Tensor, index: Optional[Tensor] = None, feature_dim: int = 1
) -> Tensor:
    if index is not None:
        return global_add_pool(x, index)
    else:
        return x.sum(dim=feature_dim)


def _softmax_aggr(
    x: Tensor,
    index: Optional[Tensor] = None,
    feature_dim: int = 1,
    t: float = 1.0,
) -> Tensor:
    """
    Softmax-weighted mean over the same axis `_sum_aggr` sums over, per channel.

    Reproduces `torch_geometric.nn.aggr.SoftmaxAggregation(learn=False, t=t)`:
    weights are a softmax of the values themselves, so they sum to one within each
    graph and each channel. The result is a weighted mean rather than a total --
    unlike `_sum_aggr`, it carries no information about how many atoms or residues
    were pooled. That difference is the whole point; see `DTIModel._probe_pool`.

    Written out rather than instantiated as a module so it stays stateless: DTIModel
    is built to load a frozen checkpoint with a strict `load_state_dict`, and an
    aggregation module would register buffers/parameters the checkpoint lacks.
    """
    if index is not None:
        # Segmented softmax down dim 0, independently per feature column.
        weight = segment_softmax(t * x, index, dim=0)
        return global_add_pool(weight * x, index)
    weight = torch.softmax(t * x, dim=feature_dim)
    return (weight * x).sum(dim=feature_dim)


class GlobalSumDecoder(Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int = 1,
        act: str = "silu",
        feature_dim: int = -1,
    ) -> None:
        super().__init__()
        self.act = resolve_act(act)
        self.f_ligand = Sequential(
            Linear(hidden_channels, hidden_channels),
            self.act,
            LayerNorm(hidden_channels),
        )
        self.f_pocket = Sequential(
            Linear(hidden_channels, hidden_channels),
            self.act,
            LayerNorm(hidden_channels),
        )
        self.f_combined = Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            self.act,
            Linear(hidden_channels, out_channels),
        )
        self.feature_dim = feature_dim

    def combined_representation(
        self,
        ligand_embeddings: Tensor,
        pocket_embeddings: Tensor,
        ligand_batch: Optional[Tensor] = None,
        pocket_batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Everything the decoder computes up to (but excluding) the prediction head:
        the per-complex vector that `f_combined` maps to an affinity.
        """
        ligand_repr = self.f_ligand(_sum_aggr(ligand_embeddings, ligand_batch))
        pocket_repr = self.f_pocket(_sum_aggr(pocket_embeddings, pocket_batch))
        return torch.cat((ligand_repr, pocket_repr), dim=self.feature_dim)

    def forward(
        self,
        ligand_embeddings: Tensor,
        pocket_embeddings: Tensor,
        ligand_batch: Optional[Tensor] = None,
        pocket_batch: Optional[Tensor] = None,
    ):
        combined_repr = self.combined_representation(
            ligand_embeddings, pocket_embeddings, ligand_batch, pocket_batch
        )
        return self.f_combined(combined_repr)


class ResidueFeaturization:
    Kissim = "kissim"
    Onehot = "onehot"


def make_model(config):
    if config.residue_featurization == ResidueFeaturization.Onehot:
        ResidueModel = ResidueTransformer
        config["residue_size"] = len(known_residues) + 1
    elif config.residue_featurization == ResidueFeaturization.Kissim:
        ResidueModel = KissimTransformer
        config["residue_size"] = 6
    else:
        raise ValueError(config.residue_featurization)
    return DTIModel(
        config,
        LigandGINE,
        ResidueModel,
        GlobalSumDecoder,
        prob=config.get("prob", True),
        emit_tower_reprs=config.get("emit_tower_reprs", False),
        repr_aggr=config.get("repr_aggr", "sum"),
        repr_aggr_t=config.get("repr_aggr_t", 1.0),
    )
