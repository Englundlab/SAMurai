"""Temporary navigation and selection shortcuts for the workflow."""
import numpy as np
from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QApplication, QWidget, QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox


class Navigation:
    def navigation_event(self, watched, event):
        if event.type() == QEvent.WindowDeactivate and watched is self.viewer.window._qt_window:
            self._pan_keys.clear()
            self.release_shift_selection()
            self.refresh_tool_highlight()
        if event.type() not in (QEvent.KeyPress, QEvent.KeyRelease, QEvent.ShortcutOverride):
            return False
        if (self.busy or not isinstance(watched, QWidget)
                or watched.window() is not self.viewer.window._qt_window
                or QApplication.activeModalWidget() is not None):
            return False
        key = event.key()
        # Always release held shortcuts, even if focus moved to a text field.
        if event.type() == QEvent.KeyRelease and not event.isAutoRepeat():
            if key in self._pan_keys:
                self._pan_keys.discard(key)
                self.refresh_tool_highlight()
                return True
            if key == Qt.Key_Shift and self._shift_layer is not None:
                self.release_shift_selection()
                return True
        if isinstance(QApplication.focusWidget(), (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return False
        directions = {Qt.Key_Left: (0, -1), Qt.Key_Right: (0, 1),
                      Qt.Key_Up: (-1, 0), Qt.Key_Down: (1, 0)}
        if key in directions and self.exists(self.image) and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)):
            if event.type() == QEvent.KeyPress:
                self._pan_keys.add(key)
                center = np.array(self.viewer.camera.center)
                center[-2:] += np.array(directions[key]) * 24 / self.viewer.camera.zoom
                self.viewer.camera.center = tuple(center)
                self.refresh_tool_highlight()
            event.accept()
            return True
        if (key == Qt.Key_Shift and event.type() == QEvent.KeyPress and not event.isAutoRepeat()
                and not self.exists(self.labels) and self.exists(self.boxes) and self.boxes.visible):
            active = self.viewer.layers.selection.active
            if active is not None and str(active.mode) == "pan_zoom":
                self._shift_layer = active
                self.viewer.layers.selection.active = self.boxes
                self.boxes.mode = "select"
                self.refresh_tool_highlight()
                return True
        return False

    def release_shift_selection(self):
        previous = self._shift_layer
        self._shift_layer = None
        if previous is not None and self.exists(self.boxes) and self.viewer.layers.selection.active is self.boxes:
            if str(self.boxes.mode) == "select":
                self.boxes.mode = "pan_zoom"
                if self.exists(previous):
                    self.viewer.layers.selection.active = previous
        self.refresh_tool_highlight()

    def middle_click_object(self, viewer, event):
        if (event.button != 3 or self.busy or not self.exists(self.labels) or not self.labels.visible
                or self.exists(self.boundary_layer) or self.exists(self.drawing_layer)):
            return
        value = self.labels.get_value(event.position, world=True)
        if value is None or int(value) <= 0:
            return
        keep_selection = (viewer.layers.selection.active is self.labels
                          and str(self.labels.mode) == "pick")
        self.viewer.layers.selection.active = self.labels
        self.labels.selected_label = int(value)
        if self.multi_select.isChecked():
            self.merge_ids.symmetric_difference_update({int(value)})
            self.labels.mode = "pick"
        else:
            self.labels.mode = "pick" if keep_selection else "pan_zoom"
        self.highlight_selected_object()
        self.refresh_tool_highlight()
