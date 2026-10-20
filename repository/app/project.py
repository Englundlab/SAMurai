"""Portable batch projects and atomic, per-image checkpoints (no GUI dependency)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from functools import wraps
from zipfile import BadZipFile

import numpy as np

from .core import read_image, illumination_correct, illumination_radius

IMAGE_EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_access(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        # Windows cannot replace a checkpoint while the GUI has it open.
        with self._checkpoint_lock:
            return method(self, *args, **kwargs)
    return locked


class Project:
    """The manifest is immutable; each checkpoint is independently replaceable."""

    def __init__(self, path):
        self._checkpoint_lock = threading.RLock()
        self.path = Path(path).resolve()
        self.root = self.path.parent
        manifest = json.loads(self.path.read_text(encoding='utf-8'))
        if manifest.get('format') != 'spheroid-project' or manifest.get('version') != 1:
            raise ValueError('Unsupported Spheroid project format.')
        self.items = manifest['images']
        if not self.items:
            raise ValueError('Project has no images.')
        for item in self.items:
            self.image_path(item)

    @classmethod
    def create(cls, folder, paths):
        folder = Path(folder).resolve()
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Choose an empty folder for the new project.')
        paths = list(dict.fromkeys(Path(p).resolve() for p in paths))
        if not paths:
            raise ValueError('Select at least one image.')
        folder.mkdir(parents=True, exist_ok=True)
        # Stage the whole import so a failed copy never publishes a partial project.
        with tempfile.TemporaryDirectory(prefix='.import_', dir=folder) as temporary:
            stage = Path(temporary)
            (stage / 'images').mkdir()
            items = []
            for index, source in enumerate(paths):
                name = f'{index:06d}_{source.name}'
                destination = stage / 'images' / name
                shutil.copy2(source, destination)
                items.append({'name': source.name, 'image': 'images/' + name,
                              'sha256': fingerprint(destination)})
            manifest = {'format': 'spheroid-project', 'version': 1, 'images': items}
            (stage / 'project.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            (stage / 'images').rename(folder / 'images')
            (stage / 'project.json').rename(folder / 'project.json')
        return cls(folder / 'project.json')

    def image_path(self, item):
        path = (self.root / item['image']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Project image must be inside the project folder.')
        return path

    def read_image(self, index):
        item = self.items[index]
        path = self.image_path(item)
        if fingerprint(path) != item['sha256']:
            raise ValueError(f"Image changed since import: {item['name']}. Import it into a new project.")
        return read_image(path)

    def checkpoint(self, index):
        return self.root / 'annotations' / f'{index:06d}.npz'

    def prepared_image(self, index, original, radius):
        """Rebuildable cache, independent of detection/review checkpoints."""
        identity = {'version': 1, 'sha256': self.items[index]['sha256'], 'radius': int(radius)}
        path = self.root / 'prepared' / f'{index:06d}.npz'
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if json.loads(str(archive['identity'].item())) == identity:
                        image = archive['image']
                        if image.shape == original.shape and np.isfinite(image).all():
                            return image.copy()
            except (ValueError, OSError, KeyError, EOFError, BadZipFile):
                pass  # A damaged cache is disposable; annotations are not.
        image = illumination_correct(original, radius)
        path.parent.mkdir(exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.prepared_', suffix='.npz', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, identity=np.array(json.dumps(identity)), image=image)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return image

    @checkpoint_access
    def load(self, index):
        path = self.checkpoint(index)
        if not path.exists():
            return {'status': 'pending'}, None
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata'].item()))
            if metadata.get('image_sha256') != self.items[index]['sha256']:
                raise ValueError('Annotation does not match the project image.')
            labels = archive['labels'].copy() if 'labels' in archive else None
        return metadata, labels

    @checkpoint_access
    def save(self, index, metadata, labels=None):
        path = self.checkpoint(index)
        path.parent.mkdir(exist_ok=True)
        metadata = dict(metadata, image_sha256=self.items[index]['sha256'])
        fd, temporary = tempfile.mkstemp(prefix='.checkpoint_', suffix='.npz', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                arrays = {'metadata': np.array(json.dumps(metadata, allow_nan=False))}
                if labels is not None:
                    arrays['labels'] = np.asarray(labels, dtype=np.uint32)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def statuses(self):
        return [item['status'] for item in self.summaries()]

    @checkpoint_access
    def summaries(self):
        result = []
        for index in range(len(self.items)):
            try:
                # Read metadata only; avoid decompressing masks to populate the list.
                path = self.checkpoint(index)
                if path.exists():
                    with np.load(path, allow_pickle=False) as archive:
                        result.append(json.loads(str(archive['metadata'].item())))
                else:
                    result.append({'status': 'pending'})
            except Exception as exc:
                result.append({'status': 'unreadable', 'error': str(exc)})
        return result


def detect_pending(project, engine, stop, progress):
    """Save each image before advancing; failures stay retryable; stop between images."""
    config = copy.deepcopy(engine.config)
    model_path = config['resolved_paths']['yolo_model']
    identity = {'filename': Path(model_path).name, 'sha256': fingerprint(model_path)}
    completed = failed = 0
    pending = [i for i, status in enumerate(project.statuses()) if status in ('pending', 'failed')]
    try:
        for position, index in enumerate(pending):
            if stop.is_set():
                break
            item = project.items[index]
            def report(stage):
                progress({'kind': 'batch', 'index': index, 'position': position + 1,
                          'total': len(pending), 'completed': completed + failed,
                          'stage': stage, 'name': item['name']})
            report('Correcting lighting')
            previous, _ = project.load(index)
            metadata = dict(previous)
            try:
                image = project.read_image(index)
                radius = illumination_radius(image)
                image = project.prepared_image(index, image, radius)
                report('Running YOLO')
                boxes, scores, counts = engine.detect(image, keep_model=True)
                prompts = {'boxes_xyxy': boxes.tolist(), 'confidence': scores.tolist(),
                           'source': ['yolo'] * len(scores), 'pending_delete': [False] * len(scores)}
                metadata.update(status='detected', boxes=prompts, original_detections=copy.deepcopy(prompts),
                                counts=counts, corrected=True, radius=radius, shape=list(image.shape),
                                detection_config=config['inference'], yolo_model=identity, error=None,
                                preprocessing_set=True)
                project.save(index, metadata)
                completed += 1
            except Exception as exc:
                metadata.update(status='failed', error=str(exc))
                # A checkpoint write failure aborts instead of pretending work was saved.
                project.save(index, metadata)
                failed += 1
            report('Saved' if metadata['status'] == 'detected' else 'Needs attention')
    finally:
        engine.release_yolo()
    return completed, failed, stop.is_set()
