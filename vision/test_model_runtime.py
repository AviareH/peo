"""Backend fallback, explicit-device safeguards, and timing synchronization."""

import unittest
from unittest.mock import MagicMock

from model_runtime import select_device, synchronize


class ModelRuntimeTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"),
                                    (False, False, "cpu")):
            with self.subTest(cuda=cuda, mps=mps):
                torch = MagicMock()
                torch.cuda.is_available.return_value = cuda
                torch.backends.mps.is_available.return_value = mps
                self.assertEqual(select_device(torch, "auto"), expected)
                self.assertEqual(select_device(torch, "cpu"), "cpu")

    def test_explicit_unavailable_accelerators_are_rejected(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        for device in ("cuda", "mps"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                select_device(torch, device)

    def test_only_selected_accelerator_is_synchronized(self):
        for device in ("cpu", "cuda", "mps"):
            with self.subTest(device=device):
                torch = MagicMock()
                synchronize(torch, device)
                self.assertEqual(torch.cuda.synchronize.call_count, int(device == "cuda"))
                self.assertEqual(torch.mps.synchronize.call_count, int(device == "mps"))


if __name__ == "__main__":
    unittest.main()
