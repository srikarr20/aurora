import importlib
import os
import pkgutil
import sys
import unittest
from pathlib import Path

import src_instrument


def src_instrument_module_names():
    package_root = Path(src_instrument.__file__).parent
    return sorted(
        module_info.name
        for module_info in pkgutil.walk_packages(
            [str(package_root)],
            prefix=f"{src_instrument.__name__}.",
        )
    )


class SrcInstrumentImportTests(unittest.TestCase):
    def test_import_every_src_instrument_module(self):
        for module_name in src_instrument_module_names():
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_imports_do_not_change_temporary_working_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            before = sorted(path.name for path in tmp_path.iterdir())
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                importlib.import_module("src_instrument")
                for module_name in src_instrument_module_names():
                    importlib.import_module(module_name)
            finally:
                os.chdir(old_cwd)

            after = sorted(path.name for path in tmp_path.iterdir())
            self.assertEqual(after, before)


class LegacyValidationIntegrityTests(unittest.TestCase):
    def test_import_run_validation_succeeds(self):
        module = importlib.import_module("src_instrument.aurora_project.run_validation")
        self.assertTrue(hasattr(module, "MissingValidationOperatorError"))

    def test_import_run_validation_creates_no_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            before = sorted(path.name for path in tmp_path.iterdir())
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                importlib.reload(
                    importlib.import_module("src_instrument.aurora_project.run_validation")
                )
            finally:
                os.chdir(old_cwd)

            after = sorted(path.name for path in tmp_path.iterdir())
            self.assertEqual(after, before)

    def test_run_validation_raises_missing_operator(self):
        module = importlib.import_module("src_instrument.aurora_project.run_validation")
        with self.assertRaises(module.MissingValidationOperatorError) as ctx:
            module.run_validation("/tmp/data", "/tmp/out")
        self.assertIn("MISSING_ORIGINAL_OPERATOR", str(ctx.exception))
        self.assertIn("optimize_operator", str(ctx.exception))

    def test_main_with_paths_raises_missing_operator(self):
        module = importlib.import_module("src_instrument.aurora_project.run_validation")
        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "run_validation.py",
                "--data-root",
                "/tmp/data",
                "--output-dir",
                "/tmp/out",
            ]
            with self.assertRaises(module.MissingValidationOperatorError) as ctx:
                module.main()
        finally:
            sys.argv = old_argv

        self.assertIn("MISSING_ORIGINAL_OPERATOR", str(ctx.exception))
        self.assertIn("different scientific algorithm", str(ctx.exception))

    def test_run_validation_does_not_reference_compute_rho_raw(self):
        module_path = (
            Path(src_instrument.__file__).parent
            / "aurora_project"
            / "run_validation.py"
        )
        self.assertNotIn("compute_rho_raw", module_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
