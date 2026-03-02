"""
Deprecated / not-yet-used runner helpers (atom-level probing, unbatching).
Kept for reference only; not imported by builds_and_runs.
"""
# Unbatching functions, NOT TESTED YET
#
# def _unbatch(self,
#              node_repr: Tensor,
#              batch_index: torch.LongTensor,
#              edge_index: Tensor = None,
#              edge_repr: Tensor = None,
#              ) -> Dict:
#     """Process node representations for a batch; pool to graph repr if graph_level."""
#     ...
#
# def _slice_edge_reprs(self, edge_repr, edge_indices) -> Dict:
#     """Atom-level: slice edge features by graph (edge order is node-based)."""
#     ...
