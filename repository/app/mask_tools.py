"""Undoable object operations and temporary polygon/line drawing tools."""
import numpy as np
from skimage.draw import polygon
from shapely.geometry import Polygon, LineString, box
from shapely.ops import split
from shapely import contains_xy
from qtpy.QtWidgets import QMessageBox


class SplitTargetError(ValueError):
    """The drawn split does not divide the selected object."""


def polyline_split_indices(data, object_id, points):
    """Extend the terminal segments and partition pixel centers without gaps."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Place at least two different points for the split.")
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]]
    if len(points) < 2:
        raise ValueError("Place at least two different points for the split.")
    reach = 4 * (np.linalg.norm(data.shape) + np.max(np.linalg.norm(points, axis=1)) + 1)
    first = points[0] - points[1]
    last = points[-1] - points[-2]
    extended = np.vstack((points[0] + first / np.linalg.norm(first) * reach,
                          points, points[-1] + last / np.linalg.norm(last) * reach))
    path = LineString(extended)
    if not path.is_simple:
        raise ValueError("The extended polyline crosses itself. Move its vertices to make a single dividing path.")
    parts = list(split(box(-1, -1, data.shape[0], data.shape[1]), path).geoms)
    if len(parts) == 1:
        raise SplitTargetError("The split does not cross the selected object.")
    if len(parts) != 2:
        raise ValueError("Adjust the polyline so it divides the image into two sides.")
    coords = np.column_stack(np.where(data == object_id))
    side = contains_xy(parts[0], coords[:, 0], coords[:, 1])
    if not side.any() or side.all():
        raise SplitTargetError("The split must cross the selected object to create two nonempty parts.")
    return tuple(coords[side].T)


class MaskTools:
    def toggle_multi_selection(self, checked):
        self.merge_ids = set()
        if self.exists(self.labels):
            self.viewer.layers.selection.active = self.labels
            self.labels.mode = "pick"
            self.highlight_selected_object()
        if checked:
            self.status.setText("Click objects to add/remove them from the selection, then Merge selected objects.")
        if hasattr(self, "export_options"):
            self.update_controls()

    def select_merge_object(self, viewer, event):
        shift = "Shift" in event.modifiers
        if ((not self.multi_select.isChecked() and not shift) or not self.exists(self.labels)
                or self.busy or self.exists(self.drawing_layer) or self.exists(self.boundary_layer)
                or viewer.layers.selection.active is not self.labels
                or str(self.labels.mode) != "pick" or event.button != 1):
            return
        value = self.labels.get_value(event.position, world=True)
        if value is not None and int(value) > 0:
            if not self.multi_select.isChecked():
                previous = int(self.labels.selected_label)
                self.multi_select.setChecked(True)
                if previous > 0:
                    self.merge_ids.add(previous)
            self.merge_ids.symmetric_difference_update({int(value)})
            self.highlight_selected_object()

    def finish_mask_operation(self, selected, message):
        self.multi_select.setChecked(False)
        self.merge_ids.clear()
        self.labels.selected_label = selected
        self.dirty = True
        self.refresh_object_ids()
        self.highlight_selected_object()
        self.viewer.layers.selection.active = self.labels
        self.labels.mode = "pick"
        self.status.setText(message + " Ctrl+Z undoes this change.")

    def merge_objects(self):
        if not self.exists(self.labels) or self.busy:
            return
        ids = sorted(self.merge_ids & set(np.unique(self.labels.data)))
        if len(ids) < 2:
            self.status.setText("Select at least two objects to merge.")
            return
        indices = np.where(np.isin(self.labels.data, ids[1:]))
        self.labels.data_setitem(indices, ids[0])
        self.finish_mask_operation(ids[0], f"Merged {len(ids)} objects into object {ids[0]}.")

    def renumber_objects(self):
        if not self.exists(self.labels) or self.busy:
            return
        data = self.labels.data
        ids = np.unique(data[data > 0])
        if np.array_equal(ids, np.arange(1, len(ids) + 1)):
            self.status.setText("Object IDs are already consecutive.")
            return
        indices = np.where(data > 0)
        values = (np.searchsorted(ids, data[indices]) + 1).astype(data.dtype)
        selected = int(self.labels.selected_label)
        selected = int(np.searchsorted(ids, selected) + 1) if selected in ids else 0
        self.labels.data_setitem(indices, values)
        self.finish_mask_operation(selected, f"Renumbered {len(ids)} objects from 1 to {len(ids)}.")

    def begin_drawing(self, kind):
        if self.busy or not self.exists(self.labels):
            return
        self.multi_select.setChecked(False)
        selected = int(self.labels.selected_label)
        if kind == "line" and (selected <= 0 or not np.any(self.labels.data == selected)):
            QMessageBox.information(self, "Select an object to split",
                "Split applies only to the selected object.\n\nClick OK, then click the mask you want to split and choose Split selected object again.",
                QMessageBox.Ok)
            self.edit_mask("pick")
            self.status.setText("Click the object you want to split, then choose Split selected object.")
            return
        self.drawing_kind, self.drawing_id = kind, selected
        self.drawing_layer = self.viewer.add_shapes(ndim=2,
            name="Draw object polygon" if kind == "polygon" else f"Split object {selected} only",
            edge_color="cyan", face_color=[0, 1, 1, .12], edge_width=2)
        self.drawing_layer.mode = "add_polygon" if kind == "polygon" else "add_polyline"
        # Replace only napari's add callback: retain its moving-vertex preview,
        # but finish on right-click instead of adding another vertex.
        callbacks = self.drawing_layer.mouse_drag_callbacks
        for index, callback in enumerate(callbacks):
            if callback.__name__ == "add_path_polygon":
                callbacks[index] = self.draw_vertex_or_finish
        self.drawing_layer.mouse_double_click_callbacks.clear()
        self.drawing_layer.mouse_double_click_callbacks.append(self.finish_drawing_shape)
        self.update_controls()
        self.status.setText("Click vertices; right-click to finish, then Apply drawing."
                            if kind == "polygon" else f"Splitting object {selected} only. Click points, then right-click to finish. Ends extend automatically; other objects stay unchanged.")

    def draw_vertex_or_finish(self, layer, event):
        if event.button == 2:
            self.finish_drawing_shape(layer, event)
        elif event.button == 1:
            from napari.layers.shapes._shapes_mouse_bindings import add_path_polygon
            add_path_polygon(layer, event)

    def finish_drawing_shape(self, layer, event=None):
        if layer._is_creating:
            layer._finish_drawing()
        if len(layer.data):
            # Direct editing permits moving vertices, without starting a second shape.
            if self.draw_vertex_or_finish in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.remove(self.draw_vertex_or_finish)
            layer.mode = "direct"
            layer.selected_data = {0}
            self.status.setText((f"Splitting object {self.drawing_id} only. " if self.drawing_kind == "line" else "Drawing finished. ")
                                + "Drag vertices to adjust, then Apply drawing or Cancel drawing.")

    def cancel_drawing(self):
        layer = self.drawing_layer
        self.drawing_layer = None
        if self.exists(layer):
            self.remove_layer(layer)
        if self.exists(self.labels):
            self.viewer.layers.selection.active = self.labels
        if hasattr(self, "export_options"):
            self.update_controls()

    def apply_drawing(self):
        try:
            if self.exists(self.drawing_layer):
                self.finish_drawing_shape(self.drawing_layer)
            if not self.exists(self.drawing_layer) or len(self.drawing_layer.data) != 1:
                raise ValueError("Draw exactly one polygon or line before applying.")
            points = np.asarray(self.drawing_layer.data[0])
            data = self.labels.data
            new_id = int(data.max()) + 1
            if new_id > np.iinfo(data.dtype).max:
                raise ValueError("No free object IDs remain. Renumber objects first.")
            if self.drawing_kind == "polygon":
                if len(points) < 3 or not Polygon(points).is_valid or Polygon(points).area == 0:
                    raise ValueError("Draw a closed polygon with at least three vertices and no crossing edges.")
                rows, cols = polygon(points[:, 0], points[:, 1], shape=data.shape)
                available = data[rows, cols] == 0
                indices = (rows[available], cols[available])
                if not len(indices[0]):
                    raise ValueError("The polygon contains no free pixels. Existing objects are preserved.")
                message = f"Created object {new_id} from the polygon."
            else:
                indices = polyline_split_indices(data, self.drawing_id, points)
                message = f"Split object {self.drawing_id} into objects {self.drawing_id} and {new_id}."
            self.labels.data_setitem(indices, new_id)
            self.cancel_drawing()
            self.finish_mask_operation(new_id, message)
        except SplitTargetError:
            object_id = self.drawing_id
            QMessageBox.information(self, "Select the object to split",
                f"This split does not divide selected object {object_id}. Split applies only to the selected object, even if the line crosses another mask.\n\nClick OK, select the intended object, then draw the split again.",
                QMessageBox.Ok)
            self.cancel_drawing()
            self.edit_mask("pick")
            self.status.setText("Select the intended object, then draw the split again.")
        except ValueError as exc:
            self.status.setText(str(exc))
