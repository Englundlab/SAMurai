import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.project import Project, detect_pending


class FakeEngine:
    def __init__(self, model, stop=None):
        self.config = {'resolved_paths': {'yolo_model': str(model)}, 'inference': {'device': 'cpu'}}
        self.calls = 0
        self.stop = stop
        self.released = False

    def detect(self, image, keep_model=False):
        assert keep_model
        self.calls += 1
        if self.stop:
            self.stop.set()
        return np.array([[1., 2., 8., 9.]]), np.array([.75]), {'candidates': 1}

    def release_yolo(self):
        self.released = True


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = []
        for i in range(3):
            path = self.root / f'{i}.png'
            Image.fromarray(np.full((12, 16), i, dtype=np.uint8)).save(path)
            self.images.append(path)
        self.model = self.root / 'model.pt'
        self.model.write_bytes(b'test identity')
        self.project = Project.create(self.root / 'project', self.images)

    def tearDown(self):
        self.temp.cleanup()

    def test_stop_resume_and_preserve_review(self):
        stop = threading.Event()
        engine = FakeEngine(self.model, stop)
        self.assertEqual(detect_pending(self.project, engine, stop, lambda m: None), (1, 0, True))
        self.assertTrue(engine.released)
        metadata, _ = self.project.load(0)
        metadata['boxes']['boxes_xyxy'][0][0] = 3
        self.project.save(0, metadata)
        stop.clear()
        next_engine = FakeEngine(self.model)
        self.assertEqual(detect_pending(self.project, next_engine, stop, lambda m: None), (2, 0, False))
        self.assertEqual(next_engine.calls, 2)
        self.assertEqual(self.project.load(0)[0]['boxes']['boxes_xyxy'][0][0], 3)
        self.assertEqual(self.project.load(0)[0]['original_detections']['boxes_xyxy'][0][0], 1)

    def test_atomic_save_failure_keeps_last_checkpoint(self):
        labels = np.arange(192, dtype=np.uint32).reshape(12, 16)
        self.project.save(0, {'status': 'segmented', 'unit': 'um'}, labels)
        with patch('app.project.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.project.save(0, {'status': 'complete'})
        metadata, loaded = self.project.load(0)
        self.assertEqual(metadata['status'], 'segmented')
        np.testing.assert_array_equal(labels, loaded)
        self.assertFalse(list(self.project.checkpoint(0).parent.glob('.checkpoint_*')))

    def test_portability_and_changed_image(self):
        moved = self.root / 'moved project'
        self.project.root.rename(moved)
        project = Project(moved / 'project.json')
        np.testing.assert_array_equal(project.read_image(1), np.ones((12, 16), dtype=np.uint8))
        project.image_path(project.items[1]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Image changed'):
            project.read_image(1)

    def test_bad_image_does_not_stop_batch(self):
        self.project.image_path(self.project.items[1]).write_bytes(b'bad')
        result = detect_pending(self.project, FakeEngine(self.model), threading.Event(), lambda m: None)
        self.assertEqual(result, (2, 1, False))
        self.assertEqual(self.project.statuses(), ['detected', 'failed', 'detected'])

    def test_zero_boxes_is_detected_and_not_repeated(self):
        engine = FakeEngine(self.model)
        engine.detect = lambda image, keep_model: (np.empty((0, 4)), np.empty(0), {'candidates': 0})
        detect_pending(self.project, engine, threading.Event(), lambda m: None)
        self.assertEqual(self.project.statuses(), ['detected'] * 3)
        self.assertEqual(detect_pending(self.project, engine, threading.Event(), lambda m: None), (0, 0, False))

    def test_batch_always_corrects_pending_images(self):
        self.project.save(0, {'status': 'pending', 'corrected': False, 'preprocessing_set': True})
        with patch('app.project.illumination_correct', side_effect=lambda image, radius: image) as correct:
            detect_pending(self.project, FakeEngine(self.model), threading.Event(), lambda m: None)
        self.assertEqual(correct.call_count, 3)
        self.assertTrue(self.project.load(0)[0]['corrected'])
        self.assertTrue(self.project.load(1)[0]['corrected'])

    def test_prepared_cache_matches_radius_and_rebuilds_damage(self):
        image = self.project.read_image(0)
        from app.core import illumination_correct
        with patch('app.project.illumination_correct', wraps=illumination_correct) as correct:
            first = self.project.prepared_image(0, image, 2)
            np.testing.assert_array_equal(first, self.project.prepared_image(0, image, 2))
            self.assertEqual(correct.call_count, 1)
            self.project.prepared_image(0, image, 3)
            self.assertEqual(correct.call_count, 2)
            (self.project.root / 'prepared' / '000000.npz').write_bytes(b'damaged')
            self.project.prepared_image(0, image, 3)
            self.assertEqual(correct.call_count, 3)

    def test_batch_radius_depends_on_each_resolution(self):
        large = self.root / 'large.png'
        Image.fromarray(np.zeros((200, 300), dtype=np.uint8)).save(large)
        project = Project.create(self.root / 'sizes', [self.images[0], large])
        detect_pending(project, FakeEngine(self.model), threading.Event(), lambda m: None)
        self.assertEqual([project.load(i)[0]['radius'] for i in range(2)], [0, 6])

    def test_engine_reuses_model_and_releases_single_image(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from app.core import Engine
        config = {'resolved_paths': {'yolo_model': str(self.model)}, 'inference': {
            'device': 'cpu', 'yolo_candidate_confidence': .15, 'yolo_imgsz': 640,
            'confidence_threshold': .25, 'containment_threshold': .9, 'nms_iou': .5}}
        xyxy, confidence = MagicMock(), MagicMock()
        xyxy.cpu.return_value.numpy.return_value = np.array([[1., 2., 8., 9.]])
        confidence.cpu.return_value.numpy.return_value = np.array([.8])
        model = MagicMock()
        model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy, conf=confidence))]
        with patch('ultralytics.YOLO', return_value=model) as constructor:
            engine = Engine(config)
            image = np.zeros((12, 16), dtype=np.uint8)
            engine.detect(image, keep_model=True)
            engine.detect(image, keep_model=True)
            self.assertEqual(constructor.call_count, 1)
            engine.release_yolo()
            engine.detect(image)
            self.assertEqual(constructor.call_count, 2)
            self.assertIsNone(engine.yolo)


if __name__ == '__main__':
    unittest.main()
