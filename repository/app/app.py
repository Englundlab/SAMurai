from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import napari
import numpy as np
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QObject, QThread, Signal, QEvent, Qt, QTimer
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSpinBox, QDoubleSpinBox, QLineEdit, QMessageBox, QScrollArea, QApplication, QCheckBox, QSizePolicy, QTabWidget, QComboBox)

from .box_history import BoxHistory, is_undo_shortcut
from .display import MASK_PALETTE, MASK_SELECTED_COLOR
from .boundary import boundary_points, smooth_rings, rasterize_boundary
from .rendering import install_rendering_fix
from .mask_tools import MaskTools
from .boxes import ReviewBoxes
from .navigation import Navigation
from .project_ui import ProjectWorkflow
from .review_queue import box_signature, mask_signature, export_target

from .core import (Engine, load_config, read_image, illumination_correct, illumination_radius, boxes_to_shapes,
                   shapes_to_boxes, export_result)


class Job(QObject):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(object)
    finished = Signal()

    def __init__(self, function):
        super().__init__()
        self.function = function

    def run(self):
        try:
            self.result.emit(self.function(self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class StatusLabel(QLabel):
    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)


class DemoPanel(ProjectWorkflow, Navigation, MaskTools, QWidget):
    def __init__(self, viewer, config):
        super().__init__()
        install_rendering_fix()
        self.viewer, self.config = viewer, config
        self.engine = Engine(config)
        self.original = self.image = self.boxes = self.labels = None
        self.path = None
        self.busy = False
        self.dirty = False
        self.corrected = False
        self.counts = {}
        self.records = []
        self.approved = None
        self.thread = None
        self._updating_features = False
        self.box_history = None
        self.boundary_layer = self.boundary_preview = None
        self.drawing_layer = None
        self.merge_ids = set()
        self._styling_boxes = False
        self._temporary_box_selection = False
        self._new_object_id = None
        self._pan_keys = set()
        self._shift_layer = None
        self.object_ids = None
        self.id_timer = QTimer(self)
        self.id_timer.setSingleShot(True)
        self.id_timer.setInterval(120)
        self.id_timer.timeout.connect(self.refresh_object_ids)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("<b>Spheroid · image review</b>")
        root_layout.addWidget(heading)
        self.instructions = QLabel("Prepare images → review boxes → refine masks → export")
        self.instructions.setWordWrap(True)
        root_layout.addWidget(self.instructions)
        self.current_image = QLabel('No image selected')
        self.current_image.setWordWrap(True)
        root_layout.addWidget(self.current_image)
        self.image_selector_layout = QVBoxLayout()
        root_layout.addLayout(self.image_selector_layout)
        self.review_navigation = QHBoxLayout()
        root_layout.addLayout(self.review_navigation)
        self.save_state = QLabel('Create a project to save and resume your work')
        self.save_state.setWordWrap(True)
        root_layout.addWidget(self.save_state)
        common_tools = QHBoxLayout()
        root_layout.addLayout(common_tools)
        self.camera_button = self.button(common_tools, 'Move camera', self.move_camera)
        self.undo_button = self.button(common_tools, 'Undo · Ctrl+Z', self.undo_edit)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)
        self.queue_status_layout = QVBoxLayout()
        root_layout.addLayout(self.queue_status_layout)
        layout = self.workflow_tab('Images')
        self.init_project_ui(layout)
        self.open_button = self.button(layout, "Open a single image…", self.open_image)
        self.model_name = QLabel("YOLO model: " + Path(config["resolved_paths"]["yolo_model"]).name)
        self.model_name.setWordWrap(True)
        layout.addWidget(self.model_name)
        self.model_button = self.button(layout, "Change YOLO model…", self.choose_yolo_model)
        self.detect_button = self.button(layout, "Run light correction + YOLO on this image", self.detect)
        self.detect_button.setToolTip('Automatically correct lighting from the original image, then detect. Replaces current boxes and masks after confirmation.')
        layout.addStretch()
        layout = self.workflow_tab('Boxes')
        self.box_guidance = QLabel('Review the boxes, then confirm to run SAM.')
        self.box_guidance.setWordWrap(True)
        layout.addWidget(self.box_guidance)
        self.show_confidence = QCheckBox("Show confidence scores")
        self.show_confidence.toggled.connect(self.toggle_confidence)
        layout.addWidget(self.show_confidence)
        note = QLabel("Review all boxes. Select/Resize/Delete.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.camera_button.setToolTip("Drag to pan; scroll to zoom. Arrow keys also pan without changing tools. Click a box edge to edit it, or hold Shift for temporary box selection.")
        self.add_button = self.button(layout, "Add manual box", self.add_box)
        self.select_button = self.button(layout, "Select / resize boxes", self.select_boxes)
        self.select_button.setToolTip("Shift+click to select multiple boxes, or drag a selection rectangle around them.")
        self.delete_selected_button = self.button(layout, "Delete selected boxes", self.delete_boxes)
        self.delete_selected_button.setToolTip("Delete the selected boxes immediately (Delete key). Ctrl+Z undoes deletion.")
        self.stage_button = self.button(layout, "Mark / unmark for deletion", self.stage_boxes)
        self.delete_button = self.button(layout, "Delete marked boxes", self.delete_staged_boxes)
        deletion_note = QLabel('Marked boxes remain included in SAM until you click Delete marked boxes.')
        deletion_note.setWordWrap(True)
        layout.addWidget(deletion_note)
        self.confirm_boxes_button = self.button(layout, 'Confirm boxes and next →', lambda: self.approve_boxes())
        self.confirm_boxes_button.setObjectName('primaryAction')
        self.sam_batch_button = self.button(layout, 'Run SAM for approved images (0)', self.start_sam_batch)
        self.segment_button = self.button(layout, "Confirm and run SAM on this image", self.segment)
        self.empty_button = self.button(layout, 'No objects here · mark reviewed and next', self.mark_empty)
        layout.addStretch()
        layout = self.workflow_tab('Masks')
        self.ready_export_mask_button = self.button(layout, 'Mark ready to export and next →', self.approve_masks)
        self.ready_export_mask_button.setObjectName('primaryAction')
        self.unmark_export_button = self.button(layout, 'Unmark ready · needs review', self.unmark_export_ready)
        mask_note = QLabel("Soft colored masks. Gold marks the selected object.")
        mask_note.setWordWrap(True)
        layout.addWidget(mask_note)
        self.pick_button = self.button(layout, "Select object", lambda: self.edit_mask("pick"))
        self.pick_button.setToolTip('Click a mask to select its ID. Middle-click also selects an object.')
        self.paint_button = self.button(layout, "Add to selected mask (brush)", lambda: self.edit_mask("paint"))
        self.erase_button = self.button(layout, "Erase mask (brush)", lambda: self.edit_mask("erase"))
        self.delete_object_button = self.button(layout, "Delete selected object", self.delete_object)
        self.delete_object_button.setToolTip("Remove all pixels with the selected object ID, including disconnected fragments. Ctrl+Z undoes deletion.")
        self.selected_object = QLabel("Select an object, then brush to add or erase pixels.")
        self.selected_object.setWordWrap(True)
        layout.addWidget(self.selected_object)
        self.show_object_ids = QCheckBox("Show object IDs")
        self.show_object_ids.setChecked(True)
        self.show_object_ids.toggled.connect(self.sync_object_id_visibility)
        layout.addWidget(self.show_object_ids)
        search_row = QHBoxLayout()
        self.object_search = QLineEdit()
        self.object_search.setPlaceholderText("Object ID")
        self.object_search.setToolTip("Enter an object ID to select it and center the view.")
        self.find_object_button = QPushButton("Find object")
        self.find_object_button.clicked.connect(self.find_object)
        self.object_search.returnPressed.connect(self.find_object)
        search_row.addWidget(self.object_search)
        search_row.addWidget(self.find_object_button)
        layout.addLayout(search_row)
        self.new_button = self.button(layout, "Draw a new object (brush)", self.new_object)
        self.mask_advanced_toggle = QCheckBox('More mask tools: polygon, merge, split, boundary')
        layout.addWidget(self.mask_advanced_toggle)
        self.mask_advanced = QWidget()
        layout.addWidget(self.mask_advanced)
        self.mask_advanced_toggle.toggled.connect(self.mask_advanced.setVisible)
        self.mask_advanced.hide()
        mask_layout = layout
        layout = QVBoxLayout(self.mask_advanced)
        layout.setContentsMargins(0, 0, 0, 0)
        self.polygon_button = self.button(layout, "Draw polygon (right-click to finish)", lambda: self.begin_drawing("polygon"))
        self.multi_select = QCheckBox("Select multiple objects")
        self.multi_select.setToolTip("Click each object to add or remove it from the merge selection. Shift+click in Select object mode also starts multiple selection. Selected objects turn gold.")
        self.multi_select.toggled.connect(self.toggle_multi_selection)
        layout.addWidget(self.multi_select)
        self.merge_button = self.button(layout, "Merge selected objects", self.merge_objects)
        self.split_button = self.button(layout, "Split selected object (right-click to finish)", lambda: self.begin_drawing("line"))
        self.split_button.setToolTip("First use Select object and click the mask to split. Only that object is changed, even if the extended polyline crosses other objects.")
        self.apply_drawing_button = self.button(layout, "Apply drawing", self.apply_drawing)
        self.cancel_drawing_button = self.button(layout, "Cancel drawing", self.cancel_drawing)
        self.renumber_button = self.button(layout, "Renumber objects", self.renumber_objects)
        self.point_spacing = QSpinBox()
        self.point_spacing.setRange(2, 100)
        self.point_spacing.setValue(10)
        self.point_spacing.setToolTip("Pixels along the boundary per control point. Smaller values give finer control. Regenerate to use a new spacing.")
        self.row(layout, "Boundary point spacing (px)", self.point_spacing)
        self.boundary_button = self.button(layout, "Edit boundary points", self.begin_boundary)
        boundary_note = QLabel("Edits the selected object's boundary only.")
        boundary_note.setWordWrap(True)
        layout.addWidget(boundary_note)
        self.boundary_button.setToolTip("Select one mask first, then edit its boundary points. Other objects remain unchanged.")
        self.apply_boundary_button = self.button(layout, "Apply smooth boundary", self.apply_boundary)
        self.cancel_boundary_button = self.button(layout, "Cancel boundary edit", self.cancel_boundary)
        mask_layout.addStretch()
        layout = self.workflow_tab('Export')
        export_note = QLabel('Measurements come from the edited masks. Set calibration and choose the files you need.')
        export_note.setWordWrap(True)
        layout.addWidget(export_note)
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(6)
        self.pixel_size.setRange(.000001, 100000)
        self.pixel_size.setValue(config["measurement"]["pixel_size"])
        self.row(layout, "Units per pixel (isotropic)", self.pixel_size)
        self.unit = QLineEdit(config["measurement"]["unit"])
        self.pixel_size.valueChanged.connect(self.mark_project_changed)
        self.unit.textChanged.connect(self.mark_project_changed)
        self.pixel_size.valueChanged.connect(self.invalidate_mask_approval)
        self.unit.textChanged.connect(self.invalidate_mask_approval)
        self.row(layout, "Measurement unit", self.unit)
        layout.addWidget(QLabel("Export formats"))
        self.export_options = {}
        for key, title, checked in (("csv", "Measurements (.csv)", True),
                                    ("xlsx", "Measurements (.xlsx)", False),
                                    ("labels", "Instance IDs (.tif)", True),
                                    ("binary", "Binary mask (.tif)", True)):
            option = QCheckBox(title)
            option.setChecked(checked)
            self.export_options[key] = option
            layout.addWidget(option)
        self.output_folder = QLineEdit(config["resolved_paths"]["outputs"])
        self.row(layout, "Output folder", self.output_folder)
        self.export_name = QLineEdit()
        self.export_name.textChanged.connect(self.mark_project_changed)
        self.row(layout, "Export subfolder", self.export_name)
        self.browse_output_button = self.button(layout, "Choose output folder…", self.choose_output_folder)
        self.export_button = self.button(layout, "Export current image", self.export)
        self.export_policy = QComboBox()
        self.export_policy.addItem('Skip existing results', False)
        self.export_policy.addItem('Replace existing results', True)
        self.row(layout, 'Batch export', self.export_policy)
        self.merge_measurements = QCheckBox('Merge all measurements into one CSV and Excel file')
        self.merge_measurements.setToolTip('For Export all images. Also saves CSV and Excel measurements in every image folder. Existing results follow the batch export policy.')
        layout.addWidget(self.merge_measurements)
        self.export_batch_button = self.button(layout, 'Export all images (0)', self.start_export_batch)
        self.export_batch_button.setObjectName('primaryAction')
        self.status = StatusLabel("Ready. Device: " + config["inference"]["device"])
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status.setFixedHeight(self.status.fontMetrics().lineSpacing() * 5 + 16)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.status.setStyleSheet("border-top: 1px solid #59616b; padding: 8px;")
        root_layout.addWidget(self.status)
        layout.addStretch()
        self.setMinimumWidth(380)
        self.setStyleSheet('QPushButton#primaryAction { background-color: #245c83; color: white; padding: 8px; } '
                          'QPushButton#primaryAction:disabled { background-color: #343c45; color: #9299a3; } '
                          'QLabel#sectionTitle { font-weight: bold; } QTabWidget::pane { border: 0; }')
        self.tool_buttons = (self.camera_button, self.add_button, self.select_button,
            self.new_button, self.polygon_button, self.pick_button, self.paint_button,
            self.erase_button, self.split_button, self.boundary_button)
        for button in self.tool_buttons:
            button.setStyleSheet('QPushButton[activeTool="true"] { background-color: #245c83; border: 2px solid #79c8ff; color: white; }')
            button.setProperty("activeTool", False)
        self.viewer.layers.events.inserted.connect(self.watch_tool_layer)
        self.viewer.layers.selection.events.active.connect(self.refresh_tool_highlight)
        for layer in self.viewer.layers:
            self.watch_tool_layer(layer=layer)
        self.viewer.layers.events.removed.connect(lambda e: self.update_controls())
        self.viewer.layers.events.removed.connect(lambda e: self.id_timer.start())
        self.viewer.mouse_drag_callbacks.append(self.select_box_edge_from_camera)
        self.viewer.mouse_drag_callbacks.append(self.leave_temporary_box_selection)
        self.viewer.mouse_drag_callbacks.append(self.select_merge_object)
        self.viewer.mouse_drag_callbacks.append(self.middle_click_object)
        QApplication.instance().installEventFilter(self)
        self.update_controls()

    def workflow_tab(self, title):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, title)
        return layout

    def eventFilter(self, watched, event):
        if self.project_close_event(watched, event):
            return True
        if self.navigation_event(watched, event):
            return True
        active = self.viewer.layers.selection.active
        can_undo = ((self.exists(self.labels) and active is self.labels and self.labels.editable)
                    or (self.box_history is not None and self.exists(self.boxes)
                        and self.boxes.editable and active is self.boxes))
        if (can_undo and not self.busy
                and isinstance(watched, QWidget)
                and watched.window() is self.viewer.window._qt_window
                and is_undo_shortcut(event, QApplication.focusWidget())):
            event.accept()
            if event.type() == QEvent.KeyPress and not event.isAutoRepeat():
                self.undo_edit()
            return True
        return super().eventFilter(watched, event)

    def undo_edit(self):
        if self.busy:
            return
        if self.exists(self.labels):
            if not self.labels.editable:
                return
            if self.labels._undo_history:
                self.viewer.layers.selection.active = self.labels
                self.labels.undo()
                self.dirty = True
                self.status.setText("Mask edit undone.")
            else:
                self.status.setText("No mask edits to undo.")
        else:
            self.undo_box_edit()

    def undo_box_edit(self):
        if self.busy or not self.exists(self.boxes) or not self.boxes.editable or self.box_history is None:
            return
        self._updating_features = True
        try:
            restored = self.box_history.undo()
        finally:
            self._updating_features = False
        if restored:
            self.refresh_box_style()
            self.dirty = True
            self.approved = None
            self.status.setText("Box edit undone. Coordinates and confidence restored.")
        else:
            self.status.setText("No box edits to undo.")

    @staticmethod
    def row(layout, label, widget):
        line = QHBoxLayout()
        line.addWidget(QLabel(label))
        line.addWidget(widget)
        layout.addLayout(line)

    @staticmethod
    def button(layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def exists(self, layer):
        return layer is not None and layer in self.viewer.layers

    def watch_tool_layer(self, event=None, layer=None):
        layer = layer if layer is not None else event.value
        layer.events.mode.connect(self.refresh_tool_highlight)
        if hasattr(layer.events, "selected_label"):
            layer.events.selected_label.connect(self.refresh_tool_highlight)

    def refresh_tool_highlight(self, event=None):
        if not hasattr(self, "tool_buttons"):
            return
        layer = self.viewer.layers.selection.active
        mode = str(layer.mode) if layer is not None else ""
        active = None
        if mode == "pan_zoom":
            active = self.camera_button
        elif layer is self.boxes and layer is not None:
            active = {"select": self.select_button, "add_rectangle": self.add_button}.get(mode)
        elif layer is self.labels and layer is not None:
            active = {"pick": self.pick_button, "erase": self.erase_button, "paint": self.paint_button}.get(mode)
            if mode == "paint" and self._new_object_id == int(layer.selected_label):
                active = self.new_button
        elif layer is self.boundary_layer and layer is not None:
            active = self.boundary_button
        elif layer is self.drawing_layer and layer is not None:
            active = self.polygon_button if self.drawing_kind == "polygon" else self.split_button
        for button in self.tool_buttons:
            selected = button is active or (button is self.camera_button and bool(self._pan_keys))
            if button.property("activeTool") != selected:
                button.setProperty("activeTool", selected)
                button.style().unpolish(button)
                button.style().polish(button)
                button.update()

    def update_controls(self):
        idle = not self.busy
        spline = self.exists(self.boundary_layer)
        drawing = self.exists(self.drawing_layer)
        boundary = spline or drawing
        self.update_project_controls(idle, boundary)
        image = self.exists(self.image)
        boxes = self.exists(self.boxes)
        labels = self.exists(self.labels)
        self.object_search.setEnabled(idle and labels and not boundary)
        self.find_object_button.setEnabled(idle and labels and not boundary)
        for widget in (self.open_button,
                       self.pixel_size, self.unit, self.output_folder, self.browse_output_button,
                       self.export_name, self.model_button, self.show_confidence, self.point_spacing,
                       *self.export_options.values()):
            widget.setEnabled(idle and not boundary)
        self.detect_button.setEnabled(idle and image and not boundary)
        if self.project is not None:
            self.open_button.setEnabled(False)
        self.mask_advanced_toggle.setEnabled(idle and labels and not boundary)
        empty = boxes and not labels and not len(self.boxes.data)
        self.empty_button.setEnabled(idle and empty and not boundary and self.project is not None)
        self.empty_button.setVisible(empty)
        statuses = [item['status'] for item in self.project_summaries]
        self.tabs.setTabEnabled(1, boxes or 'ready_sam' in statuses)
        self.tabs.setTabEnabled(2, labels)
        self.tabs.setTabEnabled(3, labels or 'ready_export' in statuses)
        self.camera_button.setEnabled(idle and image)
        for button in (self.add_button, self.select_button, self.stage_button, self.delete_button, self.delete_selected_button, self.segment_button):
            button.setEnabled(idle and boxes and image and not boundary)
        self.undo_button.setEnabled(idle and (boxes or labels) and not boundary)
        self.undo_button.setToolTip("Undo the last mask edit or object operation." if labels else "Undo the last box edit.")
        for button in (self.paint_button, self.erase_button, self.pick_button, self.new_button, self.delete_object_button, self.export_button):
            button.setEnabled(idle and labels and not boundary)
        for widget in (self.polygon_button, self.multi_select, self.merge_button, self.split_button, self.renumber_button):
            widget.setEnabled(idle and labels and not boundary)
        self.apply_drawing_button.setEnabled(idle and drawing)
        self.cancel_drawing_button.setEnabled(idle and drawing)
        self.boundary_button.setEnabled(idle and labels and not drawing)
        self.boundary_button.setText("Select boundary points" if spline else "Edit boundary points")
        self.apply_boundary_button.setEnabled(idle and spline)
        self.cancel_boundary_button.setEnabled(idle and spline)
        if self.multi_select.isChecked():
            for button in (self.delete_object_button, self.split_button, self.boundary_button):
                button.setEnabled(False)
        for layer in (self.boxes, self.labels):
            if self.exists(layer):
                editable = idle and not boundary and (layer is not self.boxes or not labels)
                if layer.editable != editable:
                    layer.editable = editable
        self.refresh_tool_highlight()

    def discard_ok(self):
        if not self.dirty:
            return True
        return QMessageBox.question(self, "Unsaved work", "This will replace the current boxes and masks. Continue without exporting?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def clear_results(self):
        self.cancel_drawing()
        self.multi_select.setChecked(False)
        self.merge_ids.clear()
        self.cancel_boundary()
        for layer in (self.boxes, self.labels):
            if self.exists(layer):
                self.remove_layer(layer)
        self.boxes = self.labels = None
        self.box_history = None
        self.approved = None
        self.records = []
        self.counts = {}
        self.dirty = False
        self.selected_object.setText("Select an object, then brush to add or erase pixels.")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open single 2D image", str(Path(self.path).parent if self.path else Path.home()),
                                             "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        if not self.discard_ok():
            return
        try:
            data = read_image(path)
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.clear_results()
        if self.exists(self.image):
            self.remove_layer(self.image)
        self.path, self.original = str(path), data
        self.export_name.setText(Path(path).stem)
        self.corrected = False
        self.image = self.viewer.add_image(data, name=Path(path).stem, rgb=data.ndim == 3)
        self.image.editable = False
        self.viewer.reset_view()
        self.tabs.setCurrentIndex(0)
        self.status.setText(f"Loaded {data.shape[1]} × {data.shape[0]}. Run YOLO; light correction is automatic.")
        self.update_controls()

    def start_job(self, function, on_result):
        self.busy = True
        self.update_controls()
        self.thread = QThread(self)
        self.job = Job(function)
        self.job.moveToThread(self.thread)
        self.thread.started.connect(self.job.run)
        self.job.progress.connect(self.on_job_progress)
        self.job.result.connect(on_result)
        self.job.failed.connect(self.show_error)
        self.job.finished.connect(self.thread.quit)
        self.job.finished.connect(self.job.deleteLater)
        self.thread.finished.connect(self.job_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def job_finished(self):
        self.busy = False
        self.thread = None
        if self.batch_running:
            # Even a fatal batch failure may follow successfully saved images.
            message = self.status.text()
            self.batch_done((0, 0, True))
            self.status.setText(message)
        elif self.project is not None:
            self.save_project_current()
        self.update_controls()
        if self._advance_after_export:
            self._advance_after_export = False
            self.next_review()
        self._export_next = False

    def show_error(self, message):
        self.status.setText("Action failed. See details and retry.")
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Spheroid demo")
        dialog.setText(message.strip().splitlines()[-1])
        dialog.setDetailedText(message)
        dialog.exec()

    def detect(self):
        if not self.discard_ok():
            return
        original = self.original.copy()
        project, index = self.project, self.project_index
        radius = illumination_radius(original)
        def prepare_and_detect(progress):
            progress('Correcting lighting automatically…')
            image = project.prepared_image(index, original, radius) if project else illumination_correct(original, radius)
            progress('Running YOLO on the corrected image…')
            return image, radius, self.engine.detect(image)
        self.status.setText('Preparing image and running YOLO…')
        self.start_job(prepare_and_detect, self.prepared_detection_done)

    def prepared_detection_done(self, result):
        image, radius, detections = result
        self.image.data = image
        self.image.contrast_limits = (0, 1)
        self.corrected, self.applied_radius = True, radius
        self.detection_done(detections)

    def choose_yolo_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose YOLO detection model", str(Path(self.config["resolved_paths"]["yolo_model"]).parent), "YOLO weights (*.pt)")
        if path:
            self.set_yolo_model(path)

    def set_yolo_model(self, path):
        if self.busy or (self.project is None and not self.discard_ok()):
            return
        def validate(progress):
            from ultralytics import YOLO
            model = YOLO(str(path))
            if model.task != "detect":
                raise ValueError("Choose a YOLO object-detection model (.pt).")
            return str(Path(path).resolve())
        self.status.setText("Checking YOLO model…")
        self.start_job(validate, self.model_changed)

    def model_changed(self, path):
        if self.project is None:
            self.clear_results()
        self.config["resolved_paths"]["yolo_model"] = path
        self.config["paths"]["yolo_model"] = path
        self.model_name.setText("YOLO model: " + Path(path).name)
        self.model_name.setToolTip(path)
        self.status.setText("YOLO model changed for future detections. Saved project results are preserved.")

    def detection_done(self, result):
        self._temporary_box_selection = False
        self.clear_results()
        boxes, scores, self.counts = result
        self.boxes = ReviewBoxes(boxes_to_shapes(boxes), ndim=2, shape_type="rectangle",
            name="Review boxes", edge_color="lime", face_color="transparent", edge_width=3,
            features={"confidence": scores, "source": np.array(["yolo"] * len(scores), dtype=object), "pending_delete": np.zeros(len(scores), dtype=bool)},
            feature_defaults={"confidence": 1.0, "source": "manual", "pending_delete": False},
            text={"string": "{confidence:.3f}", "size": 10, "scaling": True, "color": "yellow", "anchor": "upper_left", "visible": self.show_confidence.isChecked()})
        self.viewer.add_layer(self.boxes)
        self.boxes.events.data.connect(self.boxes_changed)
        self.watch_project_boxes()
        self.box_history = BoxHistory(self.boxes)
        # Native selection outline and handles use napari's highlight setting.
        from napari.settings import get_settings
        get_settings().appearance.highlight.highlight_color = [0., 0., 1., 1.]
        self.boxes.selected_data.events.items_changed.connect(lambda *args: self.refresh_box_style())
        self.boxes.events.data.connect(self.refresh_box_style)
        self.boxes.mode = "select"
        self.viewer.layers.selection.active = self.boxes
        self.dirty = True
        self.status.setText(f"{self.counts['candidates']} candidates → {len(boxes)} boxes after filtering. Add/delete/resize, then confirm.")
        self.record_project_detection()
        if not self.project_loading:
            self.update_controls()
            self.tabs.setCurrentIndex(1)

    def boxes_changed(self, event):
        if self._updating_features:
            return
        if str(getattr(event, "action", "")) == "added":
            self._updating_features = True
            try:
                features = self.boxes.features.copy()
                indices = list(event.data_indices)
                features.loc[features.index[indices], "confidence"] = 1.0
                features.loc[features.index[indices], "source"] = "manual"
                features.loc[features.index[indices], "pending_delete"] = False
                self.boxes.features = features
            finally:
                self._updating_features = False
        self.dirty = True
        self.approved = None
        if self.project is not None and not self.project_loading:
            for key in ('box_approval', 'mask_approval', 'exported_review', 'queue_error', 'queue_note'):
                self.project_metadata.pop(key, None)
        if self.exists(self.labels):
            # Export must never silently combine stale masks and newly edited boxes.
            self.remove_layer(self.labels)
            self.labels = None
            self.approved = None
            self.records = []
            self.status.setText("Boxes changed. Confirm and rerun SAM to create updated masks.")
            self.update_controls()

    def begin_box_review(self):
        if self.exists(self.labels):
            if not self.discard_ok():
                return False
            self.remove_layer(self.labels)
            self.labels = None
            self.approved = None
            self.records = []
            self.status.setText("Review boxes, then confirm and rerun SAM.")
            if self.project is not None:
                for key in ('box_approval', 'mask_approval', 'exported_review'):
                    self.project_metadata.pop(key, None)
            self.mark_project_changed()
            self.update_controls()
        return True

    def add_box(self):
        self._temporary_box_selection = False
        if not self.begin_box_review():
            return
        self.viewer.layers.selection.active = self.boxes
        self.boxes.visible = True
        # Changing current_properties also changes selected shapes in napari.
        self.boxes.selected_data = set()
        self.boxes.current_properties = {"confidence": np.array([1.0]), "source": np.array(["manual"]), "pending_delete": np.array([False])}
        self.boxes.mode = "add_rectangle"

    def select_boxes(self):
        self._temporary_box_selection = False
        if not self.begin_box_review():
            return
        self.viewer.layers.selection.active = self.boxes
        self.boxes.visible = True
        self.boxes.mode = "select"

    def select_box_edge_from_camera(self, viewer, event):
        """A stationary edge click leaves pan mode; actual drags still pan."""
        active = viewer.layers.selection.active
        if (self.busy or self.exists(self.labels) or not self.exists(self.boxes)
                or not self.boxes.visible or not self.boxes.editable
                or viewer.dims.ndisplay != 2 or active is None
                or str(active.mode) != "pan_zoom" or event.button != 1):
            return
        point = np.asarray(event.position[-2:], dtype=float)
        tolerance = 5.0 / viewer.camera.zoom
        nearest, distance = None, tolerance
        # Use segment distance instead of box interiors, preserving camera clicks
        # inside and outside the boxes. Prefer the topmost box on equal distances.
        for index, shape in enumerate(self.boxes.data):
            vertices = np.array([self.boxes.data_to_world(v)[-2:] for v in shape])
            starts, ends = vertices, np.roll(vertices, -1, axis=0)
            directions = ends - starts
            lengths = np.sum(directions ** 2, axis=1)
            fractions = np.clip(np.sum((point - starts) * directions, axis=1)
                                / np.maximum(lengths, 1e-12), 0, 1)
            d = np.min(np.linalg.norm(point - (starts + fractions[:, None] * directions), axis=1))
            if d <= distance:
                nearest, distance = index, d
        if nearest is None:
            return
        pressed = np.asarray(event.pos, dtype=float).copy()
        moved = False
        yield
        while event.type == "mouse_move":
            moved |= np.linalg.norm(np.asarray(event.pos) - pressed) > 4
            yield
        if (moved or np.linalg.norm(np.asarray(event.pos) - pressed) > 4
                or not self.exists(self.boxes) or self.exists(self.labels)):
            return
        self.viewer.layers.selection.active = self.boxes
        self.boxes.mode = "select"
        self._temporary_box_selection = True
        if "Shift" in event.modifiers:
            self.boxes.selected_data = set(self.boxes.selected_data) ^ {nearest}
        else:
            self.boxes.selected_data = {nearest}
        self.status.setText("Drag a handle to resize, or drag the box to move it. Click blank space to return to Move camera.")

    def leave_temporary_box_selection(self, viewer, event):
        layer = self.boxes
        if (not self._temporary_box_selection or self.busy or self.exists(self.labels)
                or not self.exists(layer) or viewer.layers.selection.active is not layer
                or str(layer.mode) != "select" or event.button != 1 or event.modifiers):
            return
        shape, handle = layer.get_value(event.position, world=True)
        if shape is not None or handle is not None:
            return
        pressed = np.asarray(event.pos, dtype=float).copy()
        moved = False
        yield
        while event.type == "mouse_move":
            moved |= np.linalg.norm(np.asarray(event.pos) - pressed) > 4
            yield
        if moved or np.linalg.norm(np.asarray(event.pos) - pressed) > 4:
            return
        # Let napari finish its selection-release callback before changing tools.
        def return_to_camera():
            if (self._temporary_box_selection and self.exists(layer)
                    and viewer.layers.selection.active is layer and str(layer.mode) == "select"):
                layer.selected_data = set()
                self.move_camera()
                self.status.setText("Move camera: drag to pan, or click a box edge to edit it.")
        QTimer.singleShot(0, return_to_camera)

    def move_camera(self):
        self._temporary_box_selection = False
        layer = self.viewer.layers.selection.active
        if layer is None:
            layer = self.image
            self.viewer.layers.selection.active = layer
        if layer is not None:
            layer.mode = "pan_zoom"

    def delete_boxes(self):
        if not self.begin_box_review():
            return
        self.boxes.remove_selected()

    def toggle_confidence(self):
        if self.exists(self.boxes):
            self.boxes.text.visible = self.show_confidence.isChecked()

    def refresh_box_style(self, event=None):
        if not self.exists(self.boxes) or self._styling_boxes or len(self.boxes.features) != len(self.boxes.data):
            return
        self._styling_boxes = True
        try:
            colors = np.tile([0., 1., 0., 1.], (len(self.boxes.data), 1))
            manual = self.boxes.features["source"].to_numpy() == "manual"
            colors[manual] = [170/255, 0., 1., 1.]
            marked = self.boxes.features["pending_delete"].to_numpy(dtype=bool)
            colors[marked] = [1., 0., 0., 1.]
            for i in self.boxes.selected_data:
                if i < len(colors) and not marked[i]:
                    colors[i] = [0., 0., 1., 1.]
            if not np.array_equal(self.boxes.edge_color, colors.astype(np.float32)):
                self.boxes.edge_color = colors
        finally:
            self._styling_boxes = False

    def stage_boxes(self):
        if not self.begin_box_review() or not self.boxes.selected_data:
            return
        self.box_history.checkpoint()
        features = self.boxes.features.copy()
        indices = list(self.boxes.selected_data)
        mark = not features.loc[indices, "pending_delete"].all()
        features.loc[indices, "pending_delete"] = mark
        self.boxes.features = features
        self.boxes.selected_data = set()
        self.refresh_box_style()
        self.dirty = True
        self.viewer.layers.selection.active = self.boxes
        self.move_camera()
        self.status.setText(f"{int(features.pending_delete.sum())} boxes marked red. They remain included in SAM until deleted.")

    def delete_staged_boxes(self):
        if not self.begin_box_review():
            return
        self.boxes.selected_data = set(np.flatnonzero(self.boxes.features.pending_delete.to_numpy(dtype=bool)))
        self.boxes.remove_selected()

    def approved_box_data(self):
        boxes = shapes_to_boxes(self.boxes.data, self.image.data.shape)
        return boxes, self.boxes.features.copy()

    def segment(self):
        if self.exists(self.labels) and not self.discard_ok():
            return
        try:
            if any(str(t) != "rectangle" for t in self.boxes.shape_type):
                raise ValueError("Only rectangle prompts are supported.")
            boxes, features = self.approved_box_data()
            if not len(boxes):
                raise ValueError("Add at least one box before running SAM.")
            scores = features["confidence"].to_numpy(dtype=np.float32).copy()
            sources = features["source"].astype(str).tolist()
            scores[np.array(sources) == "manual"] = 1.0
            self.approved = {"boxes_xyxy": boxes.tolist(), "confidence": scores.tolist(), "source": sources}
            if self.project is not None:
                prompts = dict(self.approved, confidence=features.confidence.tolist(), pending_delete=features.pending_delete.tolist())
                self.project_metadata['box_approval'] = box_signature(prompts)
                self.project_metadata['sam_settings'] = {
                    'model': Path(self.config['resolved_paths']['sam_model']).name,
                    'inference': dict(self.config['inference'])}
            if not self.save_project_current():
                return
            image = self.image.data.copy()
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.status.setText("Boxes confirmed. Preparing SAM…")
        self.start_job(lambda progress: self.engine.segment(image, boxes, scores, progress), self.segmentation_done)

    def segmentation_done(self, result):
        if self.project is not None and not self.project_loading:
            for key in ('mask_approval', 'exported_review', 'queue_error', 'queue_note'):
                self.project_metadata.pop(key, None)
            self.project_metadata['status'] = 'segmented'
        self.multi_select.setChecked(False)
        self.merge_ids.clear()
        if self.exists(self.labels):
            self.remove_layer(self.labels)
        data, self.records = result
        ids = np.unique(data[data > 0])
        self.mask_base_colors = {0: (0, 0, 0, 0), None: MASK_PALETTE[0]}
        self.mask_base_colors.update({int(i): MASK_PALETTE[(int(i)-1) % len(MASK_PALETTE)] for i in ids})
        self.labels = self.viewer.add_labels(data, name="Editable spheroid masks", opacity=.42,
            colormap=DirectLabelColormap(color_dict=self.mask_base_colors.copy()))
        self.labels.contour = 0
        self.labels.events.selected_label.connect(self.highlight_selected_object)
        self.labels.events.data.connect(lambda e: setattr(self, "dirty", True))
        self.labels.events.paint.connect(lambda e: setattr(self, "dirty", True))
        self.labels.events.data.connect(lambda e: self.id_timer.start())
        self.labels.events.paint.connect(lambda e: self.id_timer.start())
        # Native undo/redo refreshes the layer without emitting a paint event.
        self.labels.events.set_data.connect(lambda e: self.id_timer.start())
        self.watch_project_masks()
        self.labels.events.visible.connect(self.sync_object_id_visibility)
        self.labels.brush_size = 15
        self.boxes.visible = False
        self.viewer.layers.selection.active = self.labels
        self.labels.selected_label = int(ids[0]) if len(ids) else 1
        self.highlight_selected_object()
        self.refresh_object_ids()
        self.dirty = True
        self.status.setText(f"SAM finished: {len(np.unique(data[data > 0]))} objects. Pick an object, paint/erase boundaries, then export.")
        if not self.project_loading:
            self.update_controls()
            self.tabs.setCurrentIndex(2)

    def sync_object_id_visibility(self, event=None):
        if self.exists(self.object_ids):
            self.object_ids.visible = bool(self.exists(self.labels) and self.labels.visible
                                           and self.show_object_ids.isChecked())

    def refresh_object_ids(self):
        """Keep displayed IDs aligned with the editable instance mask."""
        if not self.exists(self.labels):
            if self.exists(self.object_ids):
                self.remove_layer(self.object_ids)
            self.object_ids = None
            return
        from skimage.measure import regionprops
        positions, ids = [], []
        for region in regionprops(self.labels.data):
            center = np.asarray(region.centroid)
            pixel = np.rint(center).astype(int)
            if self.labels.data[tuple(pixel)] != region.label:
                # Concave objects and rings can have a centroid outside the mask.
                coords = region.coords
                center = coords[np.argmin(np.sum((coords - center) ** 2, axis=1))]
            positions.append(center)
            ids.append(str(region.label))
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        features = {"object_id": np.asarray(ids, dtype=object)}
        if not self.exists(self.object_ids):
            active = self.viewer.layers.selection.active
            self.object_ids = self.viewer.add_points(positions, name="Object IDs", size=0,
                canvas_size_limits=(0, 0), face_color="transparent", border_color="transparent",
                features=features, text={"string": "{object_id}", "size": 12,
                    "scaling": False, "color": "red", "anchor": "center"})
            self.object_ids.editable = False
            self.viewer.layers.selection.active = active
        else:
            self.object_ids.data = positions
            self.object_ids.features = features
        self.object_ids.editable = False
        self.sync_object_id_visibility()

    def find_object(self):
        if (self.busy or not self.exists(self.labels) or self.exists(self.boundary_layer)
                or self.exists(self.drawing_layer)):
            return
        try:
            object_id = int(self.object_search.text().strip())
            if object_id <= 0:
                raise ValueError
        except ValueError:
            self.status.setText("Enter a positive whole-number object ID.")
            return
        coords = np.column_stack(np.where(self.labels.data == object_id))
        if not len(coords):
            self.status.setText(f"Object {object_id} was not found in the current mask.")
            return
        self.multi_select.setChecked(False)
        self.labels.visible = True
        self.viewer.layers.selection.active = self.labels
        self.labels.selected_label = object_id
        self.edit_mask("pick")
        self.highlight_selected_object()
        self.focus_object(object_id)
        self.status.setText(f"Object {object_id} found and selected.")

    def focus_object(self, object_id):
        """Center the selected mask with enough surrounding space for editing."""
        coords = np.column_stack(np.where(self.labels.data == object_id))
        if not len(coords):
            return
        low = np.asarray(self.labels.data_to_world(coords.min(axis=0)), dtype=float)[-2:]
        high = np.asarray(self.labels.data_to_world(coords.max(axis=0)), dtype=float)[-2:]
        center = (low + high) / 2
        # Leave surrounding context visible rather than filling the viewport
        # with a tiny object. Napari stores canvas size as height, width.
        extent = np.maximum(np.abs(high - low) + 1, 80)
        zoom = min(8.0, float(np.min(np.asarray(self.viewer._canvas_size) / extent)) * .4)
        self.viewer.camera.center = (0, *center)
        self.viewer.camera.zoom = zoom

    def highlight_selected_object(self, event=None):
        if not self.exists(self.labels):
            return
        selected = int(self.labels.selected_label)
        self.selected_object.setText(f"Selected object: {selected}. Brush adds pixels to this object."
                                     if selected > 0 else "Background selected. Click an object before adding pixels.")
        if selected > 0:
            self.mask_base_colors.setdefault(selected, MASK_PALETTE[(selected-1) % len(MASK_PALETTE)])
        colors = self.mask_base_colors.copy()
        if self.multi_select.isChecked():
            for object_id in self.merge_ids:
                colors[object_id] = MASK_SELECTED_COLOR
            self.selected_object.setText("Selected objects: " + (", ".join(map(str, sorted(self.merge_ids))) or "none"))
        elif self.labels.selected_label > 0:
            colors[selected] = MASK_SELECTED_COLOR
        # Setting a colormap also emits selected_label; avoid re-entering this
        # callback while still notifying napari's other selection listeners.
        with self.labels.events.selected_label.blocker(self.highlight_selected_object):
            self.labels.colormap = DirectLabelColormap(color_dict=colors)

    def edit_mask(self, mode):
        self._new_object_id = None
        if mode != "pick":
            self.multi_select.setChecked(False)
        self.viewer.layers.selection.active = self.labels
        if mode == "paint" and self.labels.selected_label == 0:
            self.labels.mode = "pick"
            self.status.setText("Click the object you want to extend, then choose Add to selected mask.")
            return
        self.labels.mode = mode
        self.refresh_tool_highlight()

    def delete_object(self):
        if self.busy or not self.exists(self.labels) or self.exists(self.boundary_layer):
            return
        label = int(self.labels.selected_label)
        if label <= 0:
            self.status.setText("Select an object to delete first.")
            return
        indices = np.where(self.labels.data == label)
        if not len(indices[0]):
            self.status.setText("The selected object has no pixels to delete.")
            return
        self.labels.data_setitem(indices, 0)
        self.dirty = True
        self.refresh_object_ids()
        self.viewer.layers.selection.active = self.labels
        self.labels.mode = "pick"
        self.status.setText(f"Object {label} deleted, including all fragments. Ctrl+Z undoes deletion.")

    def new_object(self):
        self.labels.selected_label = int(self.labels.data.max()) + 1
        self.edit_mask("paint")
        self._new_object_id = int(self.labels.selected_label)
        self.refresh_tool_highlight()

    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_folder.text())
        if folder:
            self.output_folder.setText(folder)

    def begin_boundary(self):
        self.multi_select.setChecked(False)
        if self.exists(self.boundary_layer):
            self.viewer.layers.selection.active = self.boundary_layer
            self.boundary_layer.mode = "select"
            self.focus_object(self.boundary_id)
            return
        try:
            label = int(self.labels.selected_label)
            if label <= 0:
                raise ValueError("Select an object first.")
            rings = boundary_points(self.labels.data == label, self.point_spacing.value())
            self.boundary_sizes = [len(r) for r in rings]
            self.boundary_id = label
            self.boundary_layer = self.viewer.add_points(np.concatenate(rings), name="Move boundary points", size=6,
                canvas_size_limits=(8, 16), face_color="black", border_color="black")
            self.boundary_preview = self.viewer.add_shapes(ndim=2, name="Smooth boundary preview", edge_color="#e8c477", edge_width=2, face_color="transparent")
            self.boundary_preview.editable = False
            self.boundary_layer.events.data.connect(self.update_boundary_preview)
            self.update_boundary_preview()
            self.viewer.layers.selection.active = self.boundary_layer
            self.boundary_layer.mode = "select"
            self.update_controls()
            self.focus_object(label)
            self.status.setText(f"Editing object {label} only. Drag boundary points, then Apply or Cancel.")
        except Exception as exc:
            self.cancel_boundary()
            self.show_error(str(exc))

    def update_boundary_preview(self, event=None):
        try:
            curves = smooth_rings(self.boundary_layer.data, self.boundary_sizes)
            self.boundary_preview.data = [(r, "path") for r in curves]
            self.boundary_preview.edge_color = "#e8c477"
        except ValueError as exc:
            self.boundary_preview.edge_color = "red"
            self.status.setText(str(exc))

    def apply_boundary(self):
        try:
            curves = smooth_rings(self.boundary_layer.data, self.boundary_sizes)
            mask = rasterize_boundary(curves, self.labels.data.shape)
            data = self.labels.data
            # Preserve every other object, even where a curve crosses it.
            mask &= (data == 0) | (data == self.boundary_id)
            if not mask.any():
                raise ValueError("The edited boundary contains no available pixels.")
            indices = np.where((data == self.boundary_id) | mask)
            values = np.where(mask[indices], self.boundary_id, 0).astype(data.dtype)
            self.labels.data_setitem(indices, values)
            self.dirty = True
            self.cancel_boundary()
            self.status.setText("Smooth boundary applied. Ctrl+Z undoes this edit.")
        except ValueError as exc:
            self.status.setText(str(exc))

    def cancel_boundary(self):
        for layer in (self.boundary_layer, self.boundary_preview):
            if self.exists(layer):
                self.remove_layer(layer)
        self.boundary_layer = self.boundary_preview = None
        if self.exists(self.labels):
            self.viewer.layers.selection.active = self.labels
        if hasattr(self, "export_options"):
            self.update_controls()

    def remove_layer(self, layer):
        """Dispose a layer without leaving enabled attributes of deleted buffers.

        Vispy 0.15 leaves vertex attributes enabled between shader programs. On
        Windows/NVIDIA, deleting a points/shapes layer can then crash the next
        image draw. Each program re-enables its own attributes on its next draw.
        """
        if layer is self.labels and self.exists(self.object_ids):
            overlay = self.object_ids
            self.object_ids = None
            self.remove_layer(overlay)
        from vispy.gloo import gl
        canvas = self.viewer.window._qt_viewer.canvas._scene_canvas
        canvas.set_current()
        self.viewer.layers.remove(layer)
        for index in range(int(gl.glGetParameter(gl.GL_MAX_VERTEX_ATTRIBS))):
            gl.glDisableVertexAttribArray(index)

    def export(self):
        formats = [key for key, option in self.export_options.items() if option.isChecked()]
        if not formats:
            self.status.setText("Choose at least one export format.")
            return
        if not self.output_folder.text().strip():
            self.status.setText("Choose an output folder.")
            return
        output_root = str(Path(self.output_folder.text().strip()).expanduser().resolve())
        self._exported_labels = "labels" in formats
        labels = self.labels.data.copy()
        metadata = {"config": self.config, "illumination_corrected": self.corrected,
                    "illumination_radius": getattr(self, "applied_radius", None) if self.corrected else None,
                    "approved_prompts": self.approved, "pre_edit_model_scores": self.records,
                    "detection_counts": self.counts}
        if self.project is not None:
            metadata['saved_project_provenance'] = {
                key: self.project_metadata.get(key)
                for key in ('yolo_model', 'detection_config', 'sam_settings', 'image_sha256')}
        size, unit = self.pixel_size.value(), self.unit.text().strip()
        folder_name = self.export_name.text().strip()
        if (not folder_name or folder_name in (".", "..") or any(c in folder_name for c in '<>:"/\\|?*')
                or folder_name.endswith((".", " "))):
            self.status.setText("Enter a valid export subfolder name, without path separators.")
            return
        target = Path(output_root) / folder_name
        if self.project is not None:
            try:
                export_target(self.project, output_root, folder_name)
            except ValueError as exc:
                self.status.setText(str(exc))
                return
        overwrite = False
        if target.exists():
            overwrite = QMessageBox.question(self, "Export folder already exists", "Replace this image's previous exported files? Other files will be kept.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
            if not overwrite:
                return
        self.status.setText("Exporting final edited masks and measurements…")
        self.start_job(lambda progress: export_result(labels, output_root,
                                                     self.path, size, unit, metadata, formats=formats, folder_name=folder_name, overwrite=overwrite), self.export_done)

    def export_done(self, directory):
        if self._exported_labels:
            self.dirty = False
        if self.project is not None:
            self.project_metadata['status'] = 'complete'
            self.project_metadata['exported_review'] = mask_signature(self.labels.data, self.pixel_size.value(), self.unit.text())
            for key in ('mask_approval', 'queue_error', 'queue_note'):
                self.project_metadata.pop(key, None)
            self.project_changed = False
            if not self.save_project_current():
                self._export_next = False
                return
            self._advance_after_export = self._export_next
        self.status.setText("Exported to:\n" + str(directory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "workflow_config.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config['inference']['device'] = args.device
    viewer = napari.Viewer(title="Spheroid segmentation — YOLO + SAM")
    panel = DemoPanel(viewer, config)
    viewer.window.add_dock_widget(panel, name="Spheroid workflow", area="right")
    # Keep Qt workers alive until they finish when the viewer is closed.
    from qtpy.QtWidgets import QApplication
    def shutdown():
        if panel.thread is not None:
            panel.thread.quit()
            panel.thread.wait()
    QApplication.instance().aboutToQuit.connect(shutdown)
    napari.run()


if __name__ == "__main__":
    main()
