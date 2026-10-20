"""Rectangle review layer with translation and resizing, but no rotation."""
from napari.layers import Shapes
import numpy as np
from napari.layers.shapes._shapes_constants import Box


class ReviewBoxes(Shapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._syncing_width = False
        self.events.scale_factor.connect(self._sync_edge_width)
        self.events.data.connect(self._sync_edge_width)
        self._sync_edge_width()

    def _sync_edge_width(self, event=None):
        if self._syncing_width:
            return
        if event is not None and str(getattr(event, "action", "")) in {"adding", "changing", "removing"}:
            return
        self._syncing_width = True
        try:
            # Let borders narrow when zooming out, but keep them visible and
            # cap them at three canvas pixels when zooming in.
            scale = self._normalized_scale_factor
            width = float(np.clip(3.0 / scale, 1.5, 3.0)) * scale
            self._current_edge_width = width
            if len(self.data) and not np.allclose(self.edge_width, width, rtol=1e-5):
                self.edge_width = width
                # napari 0.6.6's edge_width setter changes mesh data without
                # notifying Vispy. Without this, old widths stay on the GPU
                # until selecting a box happens to trigger a redraw.
                self.events.edge_width()
        finally:
            self._syncing_width = False

    def _outline_shapes(self):
        # Selection color is already applied to the actual edges by the panel.
        # Do not draw napari's extra thick selection/hover outline on top.
        return None, None

    @property
    def mode(self):
        return Shapes.mode.fget(self)

    @mode.setter
    def mode(self, value):
        # Direct vertex editing and affine transforms can also skew rectangles.
        if str(value) in {"direct", "transform", "vertex_insert", "vertex_remove"}:
            value = "select"
        Shapes.mode.fset(self, value)

    def _compute_vertices_and_box(self):
        vertices, face, edge, pos, width = super()._compute_vertices_and_box()
        if str(self.mode) == "select" and self._selected_box is not None and len(self.selected_data):
            vertices = self._selected_box[:Box.LEN, ::-1]
            # Keep resize handles, without painting a second border over edges.
            pos = None
        return vertices, face, edge, pos, width

    def _get_value(self, position):
        shape, vertex = super()._get_value(position)
        # napari returns index 8 for the ninth displayed handle (rotation).
        if str(self.mode) == "select" and vertex == Box.LEN:
            return None, None
        return shape, vertex
