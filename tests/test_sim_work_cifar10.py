"""CIFAR-10 loading when Windows denies access to extracted dataset files."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import pickle
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'sim_work_runner', ROOT / 'docs/sim-work/tools/run_experiment.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class CifarArchiveLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cifar = runner.datasets.CIFAR10
        self.archive = self.root / self.cifar.filename
        self.images = {True: [], False: []}
        self.labels = {True: [], False: []}
        with tarfile.open(self.archive, 'w:gz') as archive:
            for train, files in [(True, self.cifar.train_list),
                                 (False, self.cifar.test_list)]:
                for k, (name, _) in enumerate(files):
                    value = 20 + k if train else 180
                    data = np.full((2, 3072), value, dtype=np.uint8)
                    labels = [k, k + 1] if train else [8, 9]
                    self.images[train].extend(data)
                    self.labels[train].extend(labels)
                    self.add_member(archive, name, {'data': data, 'labels': labels})
            self.add_member(archive, self.cifar.meta['filename'],
                            {'label_names': [f'class{i}' for i in range(10)]})

    def add_member(self, archive, name, value):
        data = pickle.dumps(value)
        info = tarfile.TarInfo(f'{self.cifar.base_folder}/{name}')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    def args(self, train, download):
        return argparse.Namespace(
            smoke=False, case='calibrate' if train else 'fp',
            data_root=self.root, download=download, n=2,
            data_seed=314159, batch_size=2)

    def test_cached_archive_loads_quietly_without_probing_extracted_files(self):
        for train, download in [(True, True), (False, False)]:
            with self.subTest(train=train, download=download):
                output = io.StringIO()
                with patch.object(self.cifar, '__init__', side_effect=PermissionError(
                         13, 'Permission denied', 'data_batch_4')) as extracted_loader, \
                     patch.object(tarfile.TarFile, 'extractall', side_effect=AssertionError(
                         'Loading an archive must not extract it')), redirect_stdout(output):
                    data_loader, ids = runner.loader(self.args(train, download))
                    expected_ids = np.random.default_rng(314159).permutation(
                        len(self.labels[train]))[:2].tolist()
                    self.assertEqual(ids, expected_ids)
                    x, y = next(iter(data_loader))
                    self.assertEqual(y.tolist(), [self.labels[train][i] for i in ids])
                    self.assertEqual(tuple(x.shape), (2, 3, 32, 32))
                    means = torch.tensor([0.485, 0.456, 0.406])
                    stds = torch.tensor([0.229, 0.224, 0.225])
                    for batch_index, image_id in enumerate(ids):
                        value = float(self.images[train][image_id][0]) / 255
                        expected = ((value - means) / stds)[:, None, None].expand(3, 32, 32)
                        torch.testing.assert_close(x[batch_index], expected)
                    self.assertEqual(data_loader.dataset.dataset.classes[9], 'class9')
                    self.assertFalse((self.root / self.cifar.base_folder).exists())
                    extracted_loader.assert_not_called()
                    self.assertEqual(output.getvalue(), '')

    def test_invalid_archive_raises_a_read_error(self):
        self.archive.write_bytes(b'not a gzip archive')
        with patch.object(self.cifar, '__init__', side_effect=PermissionError(13, 'Permission denied')):
            with self.assertRaises(tarfile.ReadError):
                runner.loader(self.args(True, True))


if __name__ == '__main__':
    unittest.main()
