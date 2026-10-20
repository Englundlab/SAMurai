"""Approval fingerprints and sequential SAM/export queues, independent of Qt."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import export_result, write_combined_measurements


def box_signature(prompts):
    """Ignore harmless float precision changes when prompts pass through napari."""
    boxes = np.asarray(prompts['boxes_xyxy'], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(prompts['confidence'], dtype=np.float32)
    sources = prompts['source']
    deleted = prompts.get('pending_delete', [False] * len(boxes))
    if not (len(boxes) == len(scores) == len(sources) == len(deleted)):
        raise ValueError('Box metadata lengths differ.')
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError('Box coordinates/confidences must be finite.')
    payload = {'boxes': boxes.tolist(), 'confidence': scores.tolist(), 'source': list(sources),
               'pending_delete': list(deleted)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def mask_signature(labels, pixel_size, unit):
    labels = np.ascontiguousarray(labels, dtype=np.uint32)
    digest = hashlib.sha256()
    digest.update(json.dumps([list(labels.shape), float(pixel_size), unit.strip()]).encode())
    digest.update(memoryview(labels).cast('B'))
    return digest.hexdigest()


def default_export_name(project, index):
    return Path(project.items[index]['name']).stem


def resolved_export_name(project, index, metadata):
    name = default_export_name(project, index)
    saved = metadata.get('export_name', name)
    # Migrate only the exact automatic name used by previous versions.
    return name if saved == f'{index + 1:04d}_{name}' else saved


def export_target(project, root, name):
    if (not name or name in ('.', '..') or any(c in name for c in '<>:"/\\|?*') or name.endswith(('.', ' '))):
        raise ValueError('Enter a valid export subfolder name, without path separators.')
    target = (Path(root).expanduser().resolve() / name).resolve()
    protected = [project.root / p for p in ('images', 'annotations', 'prepared')]
    if (target == project.root or project.root.is_relative_to(target)
            or any(target.is_relative_to(p.resolve()) for p in protected)):
        raise ValueError('Choose an export destination outside the project’s image, annotation and preparation folders.')
    return target


def _run_queue(project, kind, stop, progress, operation, pending=None):
    required = 'ready_sam' if kind == 'sam' else 'ready_export'
    if pending is None:
        pending = [i for i, status in enumerate(project.statuses()) if status == required]
    result = {'kind': kind, 'completed': 0, 'failed': 0, 'skipped': 0, 'stopped': False}
    for position, index in enumerate(pending):
        if stop.is_set():
            result['stopped'] = True
            break
        def report(stage):
            progress({'kind': 'batch', 'queue': kind, 'index': index, 'position': position + 1,
                      'total': len(pending), 'completed': result['completed'] + result['failed'] + result['skipped'],
                      'stage': stage, 'name': project.items[index]['name']})
        report('Loading approved image')
        try:
            metadata, labels = project.load(index)
        except Exception as exc:
            result['failed'] += 1
            report('Cannot read checkpoint: ' + str(exc))
            continue  # Never overwrite an unreadable checkpoint with a partial record.
        try:
            outcome, updated, new_labels = operation(index, copy.deepcopy(metadata), labels, report)
        except Exception as exc:
            metadata['queue_error'] = {'stage': kind, 'message': str(exc)}
            # Persist the failure without changing the approval or existing masks.
            project.save(index, metadata, labels)
            result['failed'] += 1
            report('Failed: ' + str(exc))
            continue
        # Checkpoint failures are fatal: do not downgrade successful work or claim it is saved.
        project.save(index, updated, new_labels)
        result[outcome] += 1
        report('Skipped existing results' if outcome == 'skipped' else 'Saved')
    result['stopped'] = result['stopped'] or stop.is_set()
    return result


def segment_approved(project, engine, stop, progress):
    settings = {'model': Path(engine.config['resolved_paths']['sam_model']).name,
                'inference': copy.deepcopy(engine.config['inference'])}
    def segment(index, metadata, labels, report):
        prompts = metadata.get('boxes')
        if not prompts or metadata.get('box_approval') != box_signature(prompts):
            raise ValueError('Boxes changed after approval. Review and confirm them again.')
        boxes = np.asarray(prompts['boxes_xyxy'], dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            raise ValueError('No approved boxes. Review this image as empty instead.')
        image = project.read_image(index)
        if metadata.get('corrected'):
            image = project.prepared_image(index, image, metadata['radius'])
        h, w = image.shape[:2]
        if (np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > [w, h])
                or np.any(boxes[:, 2:] - boxes[:, :2] < 1)):
            raise ValueError('An approved box is outside the image or empty.')
        scores = np.asarray(prompts['confidence'], dtype=np.float32).copy()
        scores[np.asarray(prompts['source']) == 'manual'] = 1.
        labels, records = engine.segment(image, boxes, scores, report)
        metadata.update(status='segmented', records=records, sam_settings=settings,
                        approved={'boxes_xyxy': boxes.tolist(), 'confidence': scores.tolist(), 'source': prompts['source']})
        for key in ('mask_approval', 'exported_review', 'queue_error', 'queue_note'):
            metadata.pop(key, None)
        return 'completed', metadata, labels
    try:
        return _run_queue(project, 'sam', stop, progress, segment)
    finally:
        engine.release_sam()


def export_approved(project, output_root, formats, overwrite, stop, progress, all_images=False, merge=False):
    statuses = project.statuses()
    if all_images and any(s not in ('ready_export', 'complete', 'empty') for s in statuses):
        raise ValueError('Review all images and mark their masks ready to export first.')
    pending = [i for i, s in enumerate(statuses)
               if s == 'ready_export' or (all_images and s in ('complete', 'empty'))]
    if merge:
        formats = list(dict.fromkeys([*formats, 'csv', 'xlsx']))
    if not str(output_root).strip() or not formats or set(formats) - {'labels', 'binary', 'colored', 'csv', 'xlsx'}:
        raise ValueError('Choose an output folder and supported export formats.')
    root = Path(output_root).expanduser().resolve()
    # Reject collisions before any export, including with completed/unqueued images.
    names = [resolved_export_name(project, i, m) for i, m in enumerate(project.summaries())]
    duplicates = {name.casefold() for name in names if sum(other.casefold() == name.casefold() for other in names) > 1}
    def export(index, metadata, labels, report):
        empty = metadata.get('status') == 'empty'
        if empty:
            labels = np.zeros(project.read_image(index).shape[:2], dtype=np.uint32)
        approval = metadata.get('mask_approval') or metadata.get('exported_review')
        if not empty and (labels is None or approval != mask_signature(labels, metadata['pixel_size'], metadata['unit'])):
            raise ValueError('Masks or calibration changed after approval. Review and mark ready again.')
        name = names[index]
        if name.casefold() in duplicates:
            raise ValueError('Another project image uses this export subfolder. Give each image a unique export subfolder.')
        target = export_target(project, root, name)
        metadata.pop('queue_error', None)
        if target.exists() and not overwrite:
            metadata['queue_note'] = 'Skipped existing results.'
            return 'skipped', metadata, None if empty else labels
        original = project.read_image(index)
        if labels.shape != original.shape[:2]:
            raise ValueError('Mask dimensions do not match the image.')
        report('Exporting approved masks')
        payload = {'illumination_corrected': metadata.get('corrected', False),
                   'illumination_radius': metadata.get('radius'), 'approved_prompts': metadata.get('approved'),
                   'pre_edit_model_scores': metadata.get('records', []), 'detection_counts': metadata.get('counts', {}),
                   'saved_project_provenance': {k: metadata.get(k) for k in ('yolo_model', 'detection_config', 'sam_settings', 'image_sha256')}}
        directory = export_result(labels, root, project.image_path(project.items[index]),
                                  metadata['pixel_size'], metadata['unit'], payload, formats, name, overwrite)
        metadata.update(status='empty' if empty else 'complete', exported_review=mask_signature(labels, metadata['pixel_size'], metadata['unit']),
                        last_export={'directory': str(directory), 'formats': list(formats)})
        metadata['export_name'] = name
        metadata.pop('mask_approval', None)
        metadata.pop('queue_note', None)
        return 'completed', metadata, None if empty else labels
    result = _run_queue(project, 'export', stop, progress, export, pending)
    if merge:
        result['merge_note'] = 'Combined files not updated because the export stopped or an image failed.'
        if not result['failed'] and not result['stopped'] and pending:
            try:
                columns, rows = None, []
                for index in pending:
                    target = export_target(project, root, names[index])
                    if not all((target / f'measurements.{ext}').is_file() for ext in ('csv', 'xlsx')):
                        raise ValueError('An existing image folder is missing CSV or Excel measurements. Choose Replace existing results and export again.')
                    with (target / 'measurements.csv').open(newline='', encoding='utf-8') as handle:
                        reader = csv.reader(handle)
                        header = next(reader)
                        expected = ['label_id', 'area_px2', 'perimeter_px', 'centroid_x_px', 'centroid_y_px',
                                    'area', 'perimeter', 'centroid_x', 'centroid_y', 'unit']
                        if header != expected:
                            raise ValueError('Existing measurements have an incompatible format. Choose Replace existing results.')
                        columns = ['image_name', 'export_folder', *header]
                        for row in reader:
                            rows.append([project.items[index]['name'], names[index], int(row[0]),
                                         *[float(value) for value in row[1:-1]], row[-1]])
                write_combined_measurements(root, columns, rows, overwrite)
                result['merge_note'] = 'Combined CSV and Excel saved (including existing measurements for skipped images).'
            except Exception as exc:
                result['merge_note'] = 'Combined files not updated: ' + str(exc)
    return result
