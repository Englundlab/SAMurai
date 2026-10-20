"""Box review undo for the pinned napari 0.6.6 Shapes implementation."""
import numpy as np
from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox


class BoxHistory:
    def __init__(self, layer):
        self.layer = layer
        self.states = []
        self.pending = None
        self.restoring = False
        layer.events.data.connect(self.changed)

    def snapshot(self):
        return ([a.copy() for a in self.layer.data], list(self.layer.shape_type),
                self.layer.features.copy(deep=True), set(self.layer.selected_data))

    def checkpoint(self):
        self.states.append(self.snapshot())
        self.states = self.states[-100:]

    def changed(self, event):
        if self.restoring:
            return
        action = str(getattr(event, "action", ""))
        if action in ("adding", "changing", "removing"):
            if self.pending is None:
                self.pending = self.snapshot()
        elif action in ("added", "changed", "removed") and self.pending is not None:
            before = self.pending
            self.pending = None
            if len(before[0]) != len(self.layer.data) or any(
                    not np.array_equal(a, b) for a, b in zip(before[0], self.layer.data)):
                self.states.append(before)
                self.states = self.states[-100:]

    def undo(self):
        if self.pending is None and not self.states:
            return False
        state = self.pending if self.pending is not None else self.states.pop()
        self.pending = None
        self.restoring = True
        try:
            # Stop the live drag generator before restoring coordinates, otherwise
            # the next mouse move/release can reapply the cancelled resize.
            # These internals are confined here and tested against napari 0.6.6.
            for gen in list(self.layer._mouse_drag_gen.values()):
                gen.close()
                self.layer._persisted_mouse_event.pop(gen, None)
            self.layer._mouse_drag_gen.clear()
            self.layer._finish_drawing()
            self.layer.data = [(a.copy(), t) for a, t in zip(state[0], state[1])]
            self.layer.features = state[2].copy(deep=True)
            self.layer.mode = "select"
            self.layer.selected_data = state[3]
            self.layer.refresh()
        finally:
            self.restoring = False
        return True


def is_undo_shortcut(event, focus):
    """Leave ordinary text copy/undo alone in input fields."""
    if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
        return False
    return (event.type() in (QEvent.KeyPress, QEvent.ShortcutOverride)
            and event.modifiers() == Qt.ControlModifier
            and event.key() == Qt.Key_Z)
