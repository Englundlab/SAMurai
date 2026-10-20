"""Saved-project controls, separated from the existing single-image workflow."""
from __future__ import annotations

import copy
from pathlib import Path
import threading

import numpy as np
from qtpy.QtCore import QTimer, QLockFile, QEvent, Qt
from qtpy.QtWidgets import QFileDialog, QLabel, QComboBox, QProgressBar, QMessageBox

from .core import Engine, illumination_radius
from .box_history import BoxHistory
from .project import Project, detect_pending, fingerprint
from .review_queue import (box_signature, mask_signature, resolved_export_name, segment_approved, export_approved)

STATES = {'pending': 'Waiting', 'failed': 'Needs attention', 'unreadable': 'Needs attention',
          'detected': 'Boxes to review', 'confirmed': 'Boxes to review',
          'segmented': 'Masks to review', 'ready_sam': 'Ready for SAM', 'ready_export': 'Ready to export',
          'complete': 'Exported', 'empty': 'Reviewed · empty'}
REVIEW_STATES = {'detected', 'confirmed', 'segmented'}


class ProjectWorkflow:
    def init_project_ui(self, layout):
        self.project = None
        self.project_index = None
        self.project_lock = None
        self.project_loading = False
        self.project_changed = False
        self.batch_stop = threading.Event()
        self.batch_running = False
        self.queue_kind = 'yolo'
        self.project_metadata = {}
        self.project_summaries = []
        self._batch_index = None
        self._batch_stage = ''
        self._export_next = False
        self._advance_after_export = False
        self.project_title = QLabel('Start a project')
        self.project_title.setObjectName('sectionTitle')
        self.project_title.setWordWrap(True)
        layout.addWidget(self.project_title)
        intro = QLabel('Select individual images, or open a folder and select all its images (Ctrl+A on Windows, Cmd+A on Mac).')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.new_project_button = self.button(layout, 'Create project: select images', self.new_project)
        self.new_project_button.setObjectName('primaryAction')
        self.open_project_button = self.button(layout, 'Open saved project…', self.open_project)
        self.open_project_button.setObjectName('primaryAction')
        self.project_summary = QLabel('No images imported')
        self.project_summary.setWordWrap(True)
        layout.addWidget(self.project_summary)
        self.batch_button = self.button(layout, 'Run light correction + YOLO', self.start_batch)
        self.batch_button.setObjectName('primaryAction')
        self.batch_help = QLabel('Lighting is corrected automatically for each image’s resolution. '
                                 'Completed detections are saved and skipped when you resume.')
        self.batch_help.setWordWrap(True)
        layout.addWidget(self.batch_help)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat('Ready to prepare images')
        self.queue_status_layout.addWidget(self.batch_progress)
        self.stop_batch_button = self.button(self.queue_status_layout, 'Stop after current image', self.stop_batch)
        self.continue_review_button = self.button(layout, 'Continue review →', self.next_review)
        self.project_list = QComboBox()
        self.project_list.setMinimumContentsLength(18)
        self.project_list.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.project_list.activated.connect(self.select_project_image)
        self.image_selector_layout.addWidget(self.project_list)
        self.close_project_button = self.button(layout, 'Close project', self.close_project)
        self.previous_image_button = self.button(self.review_navigation, '‹ Previous', lambda: self.step_project(-1))
        self.next_image_button = self.button(self.review_navigation, 'Next ›', lambda: self.step_project(1))
        self.save_project_button = self.button(self.review_navigation, 'Save review', self.save_project_clicked)
        self.project_timer = QTimer(self)
        self.project_timer.setInterval(2000)
        self.project_timer.timeout.connect(self.autosave_project)
        self.project_timer.start()
        self.viewer.window._qt_window.installEventFilter(self)

    def update_project_controls(self, idle, boundary):
        ready = idle and not boundary
        for widget in (self.new_project_button, self.open_project_button):
            widget.setEnabled(ready)
            style_name = 'primaryAction' if self.project is None else ''
            if widget.objectName() != style_name:
                widget.setObjectName(style_name)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
        for widget in (self.project_list, self.previous_image_button, self.next_image_button,
                       self.batch_button, self.save_project_button, self.close_project_button):
            widget.setEnabled(ready and self.project is not None)
        statuses = [item['status'] for item in self.project_summaries]
        self.batch_button.setEnabled(ready and any(s in ('pending', 'failed') for s in statuses))
        self.continue_review_button.setEnabled(ready and any(s in REVIEW_STATES for s in statuses))
        self.previous_image_button.setEnabled(ready and self.project_index is not None and self.project_index > 0)
        self.next_image_button.setEnabled(ready and self.project_index is not None and self.project_index < len(statuses) - 1)
        self.stop_batch_button.setEnabled(self.batch_running and not self.batch_stop.is_set())
        self.stop_batch_button.setVisible(self.batch_running)
        # Opening a loose image must not accidentally save it over a project entry.
        self.open_button.setEnabled(ready and self.project is None)
        self.open_button.setVisible(self.project is None)
        self.refresh_current_image()
        self.refresh_queue_controls()

    def refresh_queue_controls(self):
        if not hasattr(self, 'export_batch_button'):
            return
        ready = not self.busy and not self.exists(self.boundary_layer) and not self.exists(self.drawing_layer)
        statuses = [item['status'] for item in self.project_summaries]
        sam_count = statuses.count('ready_sam')
        export_count = sum(s in ('ready_export', 'complete', 'empty') for s in statuses)
        self.sam_batch_button.setText(f'Run SAM for approved images ({sam_count})')
        self.export_batch_button.setText(f'Export all images ({export_count})')
        self.sam_batch_button.setEnabled(ready and sam_count > 0)
        self.export_batch_button.setEnabled(ready and export_count > 0)
        self.export_policy.setEnabled(ready)
        self.merge_measurements.setEnabled(ready and self.project is not None)
        self.confirm_boxes_button.setEnabled(ready and self.project is not None and self.exists(self.boxes)
                                             and len(self.boxes.data) > 0 and not self.exists(self.labels))
        masks = ready and self.project is not None and self.exists(self.labels)
        self.ready_export_mask_button.setEnabled(masks)
        self.unmark_export_button.setEnabled(masks and self.project_metadata.get('status') == 'ready_export')

    def mark_project_changed(self, event=None):
        if self.project is not None and not self.project_loading:
            self.project_changed = True
            self.save_state.setText('Unsaved edits · autosave when idle')

    def watch_project_boxes(self):
        for name in ('data', 'features', 'set_data'):
            getattr(self.boxes.events, name).connect(self.mark_project_changed)
        self.mark_project_changed()

    def watch_project_masks(self):
        for name in ('data', 'paint', 'set_data'):
            getattr(self.labels.events, name).connect(self.mark_project_changed)
        self.labels.events.data.connect(self.invalidate_mask_approval)
        self.labels.events.paint.connect(self.invalidate_mask_approval)
        self.mark_project_changed()

    def invalidate_mask_approval(self, event=None):
        if self.project is not None and not self.project_loading and self.exists(self.labels):
            for key in ('mask_approval', 'exported_review', 'queue_error', 'queue_note'):
                self.project_metadata.pop(key, None)
            self.project_metadata['status'] = 'segmented'
            self.mark_project_changed()
            # Update the visible flag immediately; autosave persists the pixels when idle.
            index = self.project_index
            if index is not None and index < len(self.project_summaries):
                self.project_summaries[index] = dict(self.project_metadata)
                self.project_list.setItemText(index, f'{index + 1}. {self.project.items[index]["name"]} · {STATES["segmented"]}')
                self.project_list.setItemData(index, self.project.items[index]['name'], Qt.ToolTipRole)
                self.refresh_queue_controls()

    def unmark_export_ready(self):
        if self.busy or self.project_metadata.get('status') != 'ready_export':
            return
        if self.exists(self.boundary_layer) or self.exists(self.drawing_layer):
            return
        self.invalidate_mask_approval()
        if self.save_project_current():
            self.status.setText('Approval removed. This image needs mask review before batch export.')

    def project_close_event(self, watched, event):
        if event.type() != QEvent.Close or watched is not getattr(self.viewer.window, '_qt_window', None):
            return False
        if self.busy:
            self.stop_batch()
            self.status.setText('Wait for the current image/job to finish before closing. Completed images are saved.')
            event.ignore()
            return True
        if self.exists(self.boundary_layer) or self.exists(self.drawing_layer):
            self.status.setText('Apply or cancel the current boundary/drawing edit before closing.')
            event.ignore()
            return True
        if not self.save_project_current():
            event.ignore()
            return True
        self.project_timer.stop()
        if self.project_lock:
            self.project_lock.unlock()
            self.project_lock = None
        return False

    def new_project(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Create project — step 1: select images', '',
            'Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp)')
        if not paths:
            return
        answer = QMessageBox.information(
            self, 'Images selected — choose where to save',
            f'{len(paths)} image(s) selected.\n\nNext, choose where to store your project and give the new project folder a name. '
            'The images will be copied there, and your review progress will be saved inside.\n\nClick OK to choose the project location.',
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if answer != QMessageBox.Ok:
            return
        folder, _ = QFileDialog.getSaveFileName(self, 'Create project — step 2: choose location and name the project folder',
                                               str(Path.home() / 'Spheroid project'), options=QFileDialog.ShowDirsOnly)
        if not folder or not self.save_project_current() or (self.project is None and not self.discard_ok()):
            return
        self.status.setText('Copying images into the new project…')
        self.start_job(lambda progress: Project.create(folder, paths), lambda project: self.activate_project(project, discard_checked=True))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Spheroid project', '', 'Spheroid project (project.json)')
        if path:
            try:
                self.activate_project(Project(path))
            except Exception as exc:
                self.show_error(str(exc))

    def activate_project(self, project, discard_checked=False):
        if self.project and self.project.path == project.path:
            return
        if not self.save_project_current() or (self.project is None and not discard_checked and not self.discard_ok()):
            return
        lock = QLockFile(str(project.root / '.project.lock'))
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            self.show_error('This project is already open in another application. Close it there first.')
            return
        if self.project_lock:
            self.project_lock.unlock()
        self.project_lock = lock
        self.project = project
        self.project_index = None
        self.project_metadata = {}
        self.project_loading = True
        try:
            self.clear_results()
            if self.exists(self.image):
                self.remove_layer(self.image)
            self.image = self.original = None
            self.path = None
            self.project_changed = False
        finally:
            self.project_loading = False
        self.refresh_project_list()
        statuses = project.statuses()
        index = next((i for i, status in enumerate(statuses) if status == 'detected'), 0)
        self.select_project_image(index)
        self.tabs.setCurrentIndex(0)

    def refresh_project_list(self):
        if self.project is None:
            return
        self.project_summaries = self.project.summaries()
        statuses = [item['status'] for item in self.project_summaries]
        self.project_list.blockSignals(True)
        self.project_list.clear()
        for i, (item, metadata) in enumerate(zip(self.project.items, self.project_summaries)):
            status = metadata['status']
            text = self._batch_stage if self.batch_running and self._batch_index == i else STATES.get(status, status)
            if metadata.get('queue_error'):
                text += ' · failed'
            elif metadata.get('queue_note'):
                text += ' · skipped'
            self.project_list.addItem(f'{i + 1}. {item["name"]} · {text}')
            self.project_list.setItemData(i, metadata.get('queue_error', {}).get('message') or metadata.get('queue_note') or metadata.get('error') or item['name'], Qt.ToolTipRole)
        self.restore_project_selection()
        self.project_list.blockSignals(False)
        self.project_title.setText(self.project.root.name)
        self.project_summary.setText(f'{len(statuses)} images · {statuses.count("pending")} waiting · '
                                     f'{sum(s in REVIEW_STATES for s in statuses)} to review · '
                                     f'{statuses.count("ready_sam")} ready for SAM · {statuses.count("ready_export")} ready to export')
        self.project_title.setToolTip(str(self.project.path))
        if not self.batch_running and self.queue_kind == 'yolo':
            prepared = sum(s in REVIEW_STATES or s in ('complete', 'empty', 'ready_sam', 'ready_export') for s in statuses)
            self.batch_progress.setRange(0, max(1, len(statuses)))
            self.batch_progress.setValue(prepared)
            self.batch_progress.setFormat(f'{prepared}/{len(statuses)} images prepared')
            self.batch_button.setText('Run light correction + YOLO' if any(s in ('pending', 'failed') for s in statuses)
                                      else 'All images prepared' if prepared == len(statuses) else 'Check images needing attention')
        self.refresh_queue_controls()

    def restore_project_selection(self):
        self.project_list.setCurrentIndex(self.project_index if self.project_index is not None else -1)

    def refresh_current_image(self):
        if self.path is None:
            self.current_image.setText('No image selected')
            self.save_state.setText('Create a project to save and resume your work')
            return
        name = self.project.items[self.project_index]['name'] if self.project and self.project_index is not None else Path(self.path).name
        self.current_image.setText(name)
        self.current_image.setToolTip(self.path)
        if self.project is None:
            self.save_state.setText('Single image · export to keep results')
        elif not self.project_changed:
            self.save_state.setText('Saved in project')
        if self.exists(self.boxes):
            count = len(self.boxes.data)
            legacy = ' These saved boxes used the original image. Re-run detection to use automatic correction.' if not self.corrected else ''
            guidance = ('Boxes approved. Run the SAM queue when ready. Editing boxes requires approval again.'
                        if self.project_metadata.get('status') == 'ready_sam' and not self.project_changed
                        else f'{count} {"box" if count == 1 else "boxes"}. Review, then confirm for the SAM queue or run SAM now.')
            self.box_guidance.setText(guidance + legacy)

    def next_review(self, states=None, fallback_to_first=False):
        if self.project is None or self.busy or not self.save_project_current():
            return
        count = len(self.project.items)
        start = self.project_index if self.project_index is not None else -1
        statuses = self.project.statuses()
        for offset in range(1, count + 1):
            index = (start + offset) % count
            if statuses[index] in (states or REVIEW_STATES):
                self.select_project_image(index)
                return
        if fallback_to_first and count:
            self.select_project_image(0)
            self.status.setText('Returned to the first image. Open Export when you are ready to export all images.')
            return
        self.tabs.setCurrentIndex(0)
        self.status.setText('No remaining images for this review step. Approved images are ready for their batch action.')

    def select_project_image(self, index):
        if self.project is None or not 0 <= index < len(self.project.items):
            return
        if not self.save_project_current():
            self.project_list.setCurrentIndex(self.project_index if self.project_index is not None else -1)
            return
        try:
            original = self.project.read_image(index)
            metadata, labels = self.project.load(index)
            image = self.project.prepared_image(index, original, metadata['radius']) if metadata.get('corrected') else original
            if metadata.get('shape') and list(image.shape) != metadata['shape']:
                raise ValueError('Saved annotations have different image dimensions.')
            if labels is not None and labels.shape != image.shape[:2]:
                raise ValueError('Saved masks have different image dimensions.')
        except Exception as exc:
            self.show_error(str(exc))
            self.project_list.setCurrentIndex(self.project_index if self.project_index is not None else -1)
            self.update_controls()
            return
        self.project_loading = True
        try:
            self.dirty = False
            self.clear_results()
            if self.exists(self.image):
                self.remove_layer(self.image)
            self.project_index = index
            self.project_metadata = metadata
            self.path = str(self.project.image_path(self.project.items[index]))
            self.original = original
            self.corrected = metadata.get('corrected', False)
            self.applied_radius = metadata.get('radius', original.shape[0] * original.shape[1] // 10000)
            self.image = self.viewer.add_image(image, name=self.project.items[index]['name'], rgb=image.ndim == 3)
            self.image.editable = False
            self.export_name.setText(resolved_export_name(self.project, index, metadata))
            self.pixel_size.setValue(metadata.get('pixel_size', self.config['measurement']['pixel_size']))
            self.unit.setText(metadata.get('unit', self.config['measurement']['unit']))
            prompts = metadata.get('boxes')
            if prompts is not None:
                self.detection_done((np.asarray(prompts['boxes_xyxy'], dtype=np.float32).reshape(-1, 4),
                                     np.asarray(prompts['confidence'], dtype=np.float32), metadata.get('counts', {'candidates': 0})))
                features = self.boxes.features.copy()
                features['source'] = prompts['source']
                features['pending_delete'] = prompts['pending_delete']
                self.boxes.features = features
                self.box_history = BoxHistory(self.boxes)
                self.refresh_box_style()
                self.approved = metadata.get('approved')
                if labels is not None:
                    self.segmentation_done((labels, metadata.get('records', [])))
            self.dirty = False
            self.project_changed = False
            self.viewer.reset_view()
            self.refresh_project_list()
            self.status.setText(f"Opened {self.project.items[index]['name']}: {STATES.get(metadata['status'], metadata['status'])}. "
                                + (metadata.get('queue_error', {}).get('message') or metadata.get('queue_note') or metadata.get('error') or 'Review is saved automatically.'))
            self.tabs.setCurrentIndex(2 if labels is not None else (1 if prompts is not None else 0))
        finally:
            self.project_loading = False
            self.update_controls()

    def step_project(self, step):
        if self.project_index is not None:
            self.select_project_image(self.project_index + step)

    def save_project_current(self):
        if self.project is None or self.project_index is None or self.project_loading:
            return True
        try:
            metadata = copy.deepcopy(self.project_metadata)
            metadata['image_sha256'] = self.project.items[self.project_index]['sha256']
            metadata.update(corrected=self.corrected, radius=self.applied_radius
                            if self.corrected else illumination_radius(self.original), shape=list(self.image.data.shape),
                            pixel_size=self.pixel_size.value(), unit=self.unit.text(), counts=self.counts,
                            approved=self.approved, records=self.records, export_name=self.export_name.text().strip())
            labels = self.labels.data if self.exists(self.labels) else None
            if self.exists(self.boxes):
                boxes, features = self.approved_box_data()
                metadata['boxes'] = {'boxes_xyxy': boxes.tolist(), 'confidence': features.confidence.tolist(),
                                     'source': features.source.tolist(), 'pending_delete': features.pending_delete.tolist()}
                if labels is not None:
                    signature = mask_signature(labels, metadata['pixel_size'], metadata['unit'])
                    if metadata.get('mask_approval') == signature:
                        metadata['status'] = 'ready_export'
                    elif metadata.get('exported_review') == signature:
                        metadata['status'] = 'complete'
                    elif metadata.get('status') == 'complete' and not self.project_changed and 'exported_review' not in metadata:
                        metadata['exported_review'] = signature  # migrate old exported checkpoints
                    else:
                        metadata['status'] = 'segmented'
                        for key in ('mask_approval', 'exported_review', 'queue_error', 'queue_note'):
                            metadata.pop(key, None)
                else:
                    if metadata.get('box_approval') == box_signature(metadata['boxes']) and len(boxes):
                        metadata['status'] = 'ready_sam'
                    else:
                        metadata['status'] = 'empty' if metadata.get('status') == 'empty' and not len(boxes) else 'detected'
                        for key in ('box_approval', 'queue_error', 'queue_note'):
                            metadata.pop(key, None)
                    metadata.pop('mask_approval', None)
                    metadata.pop('exported_review', None)
                metadata.pop('error', None)
            else:
                metadata.pop('boxes', None)
                metadata['status'] = metadata.get('status') if metadata.get('status') == 'failed' else 'pending'
            self.project.save(self.project_index, metadata, labels)
            self.project_metadata = metadata
            self.project_changed = False
            self.refresh_project_list()
            self.save_state.setText('Saved in project')
            return True
        except Exception as exc:
            self.project_changed = True
            self.save_state.setText('Not saved — retry Save review')
            self.status.setText('Project save failed; keep the app open and retry Save current review. ' + str(exc))
            return False

    def autosave_project(self):
        if self.project_changed and not self.busy and not self.exists(self.boundary_layer) and not self.exists(self.drawing_layer):
            self.save_project_current()

    def save_project_clicked(self):
        if self.save_project_current():
            self.status.setText('Current review saved in the project.')

    def close_project(self):
        if not self.save_project_current():
            return
        if self.project_lock:
            self.project_lock.unlock()
        self.project_lock = None
        self.project = None
        self.project_index = None
        self.project_metadata = {}
        self.project_summaries = []
        self.project_changed = False
        self.project_list.clear()
        self.project_title.setText('No batch project open')
        self.project_summary.setText('Import images or open a saved project')
        self.batch_button.setText('Run light correction + YOLO')
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat('Ready to prepare images')
        self.clear_results()
        self.tabs.setCurrentIndex(0)
        self.update_controls()

    def start_batch(self):
        if self.project is None or self.busy:
            return
        if not self.save_project_current():
            return
        self.batch_stop.clear()
        self.batch_running = True
        self.queue_kind = 'yolo'
        self.engine.release_sam()
        engine = Engine(copy.deepcopy(self.config))
        self.tabs.setCurrentIndex(0)
        self.batch_progress.setRange(0, 0)
        self.status.setText('Preparing images. Light correction and YOLO run automatically; keep the application open.')
        self.start_job(lambda progress: detect_pending(self.project, engine, self.batch_stop, progress), self.batch_done)

    def on_job_progress(self, message):
        if isinstance(message, dict) and message.get('kind') == 'batch':
            self._batch_index, self._batch_stage = message['index'], message['stage']
            self.batch_progress.setRange(0, message['total'])
            self.batch_progress.setValue(message['completed'])
            self.batch_progress.setFormat(f'{message["completed"]}/{message["total"]} images processed')
            self.status.setText(f'{message["position"]}/{message["total"]}: {message["name"]}\n{message["stage"]}')
            self.refresh_project_list()
        else:
            self.status.setText(str(message))

    def stop_batch(self):
        if self.batch_running:
            self.batch_stop.set()
            self.stop_batch_button.setEnabled(False)
            self.status.setText('Stopping after the current image has been saved…')

    def batch_done(self, result):
        self.batch_running = False
        self._batch_index = None
        # Do not overwrite fresh worker results with the pre-batch viewer state.
        index = self.project_index if self.project_index is not None else 0
        statuses = self.project.statuses()
        advance_tab = None
        if isinstance(result, dict):
            if (self.queue_kind == 'sam' and not result['stopped'] and not result['failed']
                    and all(s in ('segmented', 'ready_export', 'complete', 'empty') for s in statuses)
                    and 'segmented' in statuses):
                index = statuses.index('segmented')
                advance_tab = 2
        elif (not result[2] and not result[1]
              and not any(s in ('pending', 'failed', 'unreadable') for s in statuses)):
            index = next((i for i, s in enumerate(statuses) if s in ('detected', 'confirmed')), index)
            advance_tab = 1
        self.project_index = None
        self.select_project_image(index)
        if isinstance(result, dict):
            completed, failed, stopped = result['completed'], result['failed'], result['stopped']
            skipped = result['skipped']
            self.tabs.setCurrentIndex(advance_tab if advance_tab is not None else (1 if self.queue_kind == 'sam' else 3))
            self.status.setText(f'{"SAM" if self.queue_kind == "sam" else "Export"} batch {"stopped" if stopped else "finished"}: '
                                f'{completed} completed, {failed} failed, {skipped} skipped. ' + result.get('merge_note', 'Results saved.'))
        else:
            self.tabs.setCurrentIndex(advance_tab if advance_tab is not None else 0)
            completed, failed, stopped = result
            self.status.setText(f'Batch {"stopped" if stopped else "finished"}: {completed} detected, {failed} failed. '
                                'Results saved. Select an image to review boxes and run SAM.')

    def approve_boxes(self, advance=True):
        if self.project is None or not self.exists(self.boxes) or self.exists(self.labels):
            return False
        try:
            boxes, features = self.approved_box_data()
            if not len(boxes):
                raise ValueError('Add boxes, or mark this image as empty.')
            scores = features.confidence.to_numpy(dtype=np.float32).copy()
            sources = features.source.tolist()
            scores[np.asarray(sources) == 'manual'] = 1.
            self.approved = {'boxes_xyxy': boxes.tolist(), 'confidence': scores.tolist(), 'source': sources}
            if not self.save_project_current():
                return False
            self.project_metadata['box_approval'] = box_signature(self.project_metadata['boxes'])
            self.project_metadata.pop('queue_error', None)
            self.project_metadata.pop('queue_note', None)
            if not self.save_project_current():
                return False
        except Exception as exc:
            self.show_error(str(exc))
            return False
        if advance:
            self.next_review({'detected', 'confirmed'})
            if self.tabs.currentIndex() == 0:
                self.tabs.setCurrentIndex(1)
        return True

    def approve_masks(self):
        if self.project is None or not self.exists(self.labels):
            return
        if not self.unit.text().strip():
            self.status.setText('Enter a measurement unit in Export before approving masks.')
            return
        if not self.save_project_current():
            return
        self.project_metadata['mask_approval'] = mask_signature(self.labels.data, self.pixel_size.value(), self.unit.text())
        self.project_metadata.pop('queue_error', None)
        self.project_metadata.pop('queue_note', None)
        if self.save_project_current():
            if all(s in ('ready_export', 'complete', 'empty') for s in self.project.statuses()):
                self.tabs.setCurrentIndex(3)
                self.status.setText('All images reviewed. Choose your export settings, then export all images.')
            else:
                self.next_review({'segmented'}, fallback_to_first=True)

    def start_sam_batch(self):
        if self.project is None or self.busy or not self.save_project_current():
            return
        if 'ready_sam' not in self.project.statuses():
            self.status.setText('Confirm boxes before running the SAM queue.')
            return
        self.queue_kind = 'sam'
        self.batch_stop.clear()
        self.batch_running = True
        self.engine.release_sam()
        engine = Engine(copy.deepcopy(self.config))
        self.batch_progress.setRange(0, 0)
        self.status.setText('Running SAM for approved images. Each result will be saved before the next image.')
        self.start_job(lambda progress: segment_approved(self.project, engine, self.batch_stop, progress), self.batch_done)

    def start_export_batch(self):
        if self.project is None or self.busy or not self.save_project_current():
            return
        if any(s not in ('ready_export', 'complete', 'empty') for s in self.project.statuses()):
            self.status.setText('Review all images and mark their masks ready to export first.')
            return
        formats = [key for key, option in self.export_options.items() if option.isChecked()]
        merge = self.merge_measurements.isChecked()
        if merge:
            formats = list(dict.fromkeys([*formats, 'csv', 'xlsx']))
        root = self.output_folder.text().strip()
        if not formats or not root:
            self.status.setText('Choose an output folder and at least one export format.')
            return
        overwrite = bool(self.export_policy.currentData())
        self.queue_kind = 'export'
        self.batch_stop.clear()
        self.batch_running = True
        self.batch_progress.setRange(0, 0)
        self.status.setText('Exporting all images sequentially…')
        self.start_job(lambda progress: export_approved(self.project, root, formats, overwrite, self.batch_stop, progress,
                                                       all_images=True, merge=merge), self.batch_done)

    def mark_empty(self):
        if self.project is None or not self.exists(self.boxes) or len(self.boxes.data) or self.exists(self.labels):
            return
        self.project_metadata['status'] = 'empty'
        if self.save_project_current():
            self.next_review()

    def export_and_next(self):
        self._export_next = True
        self.export()
        if not self.busy:
            self._export_next = False

    def record_project_detection(self):
        if self.project is None or self.project_loading:
            return
        self.project_metadata = {'status': 'detected', 'preprocessing_set': True,
                                 'detection_config': copy.deepcopy(self.config['inference']),
                                 'yolo_model': {'filename': Path(self.config['resolved_paths']['yolo_model']).name,
                                                'sha256': fingerprint(self.config['resolved_paths']['yolo_model'])}}
        self.mark_project_changed()
        if self.save_project_current():
            self.project_metadata['original_detections'] = copy.deepcopy(self.project_metadata['boxes'])
            self.save_project_current()
