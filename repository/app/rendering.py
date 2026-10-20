"""Compatibility fixes for the demo's pinned napari/Vispy renderer."""
import sys
from functools import wraps


def install_rendering_fix():
    """Clear vertex inputs after each Vispy 0.15 draw on Windows.

    Vispy enables a program's vertex attributes in _pre_draw but leaves them
    enabled in _post_draw. A later image program uses fewer attributes. If a
    points/selection buffer was replaced or deleted, unused enabled inputs can
    then reference released storage and trigger a native NVIDIA access violation.
    Cleaning up per draw also covers edits and native napari layer deletion.
    This changes only the current process, not installed dependency files.
    """
    import vispy
    from vispy.gloo import gl
    from vispy.gloo.glir import GlirProgram

    if sys.platform != "win32" or not vispy.__version__.startswith("0.15."):
        return
    original = GlirProgram._post_draw
    if getattr(original, "_spheroid_cleanup", False):
        return

    @wraps(original)
    def post_draw(program):
        try:
            return original(program)
        finally:
            for _, handle, _, _, _ in program._attributes.values():
                gl.glDisableVertexAttribArray(handle)

    post_draw._spheroid_cleanup = True
    GlirProgram._post_draw = post_draw
