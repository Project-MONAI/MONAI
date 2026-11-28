# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
import unittest
import warnings
from contextlib import ExitStack
from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

from parameterized import parameterized

import monai.apps
from monai.apps import download_mmar, get_model_spec, load_from_mmar
from monai.apps.mmars import MODEL_DESC, mmars, model_desc
from monai.utils import DeprecatedError, deprecated


class TestMMARDownload(unittest.TestCase):
    """Check MMAR compatibility, deprecation, and disabled model loading offline."""

    @parameterized.expand([(0,), (f" {MODEL_DESC[0]['id'].upper()} ",)])
    def test_get_model_spec(self, item):
        """Resolve numeric indices and normalized model IDs to the original descriptor."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.assertIs(get_model_spec(item), MODEL_DESC[0])

    def test_get_model_spec_unknown(self):
        """Leave unrecognized model IDs unchanged."""
        item = "unknown_model"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.assertEqual(get_model_spec(item), item)

    def test_unique(self):
        """Keep every model descriptor addressable by a unique ID."""
        keys = [model["id"] for model in MODEL_DESC]
        self.assertEqual(len(keys), len(set(keys)))

    @parameterized.expand([(-1, "3"), (2, "2")])
    def test_download_api(self, version, expected_version):
        """Select the requested NGC version and pass its archive details to the downloader."""
        mmar_dir = Path("mmars")
        model = "test_model"
        model_dict = {f"nvidia/med/{model}": {"name": model, "latest": "3"}}
        with (
            patch.object(mmars, "_get_all_ngc_models", return_value=model_dict) as query,
            patch.object(mmars, "download_and_extract") as download,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            result = download_mmar(model, mmar_dir=mmar_dir, progress=False, version=version)

        self.assertEqual(result, mmar_dir / model)
        query.assert_called_once_with(model)
        download.assert_called_once_with(
            url=f"https://api.ngc.nvidia.com/v2/models/nvidia/med/{model}/versions/{expected_version}/zip",
            filepath=mmar_dir / f"{model}_{expected_version}.zip",
            output_dir=mmar_dir / model,
            hash_val=None,
            hash_type="md5",
            file_type="zip",
            has_base=False,
            progress=False,
        )

    @parameterized.expand([(-1, "1"), (2, "2")])
    def test_download_model_spec(self, version, expected_version):
        """Download from a descriptor without querying NGC, honoring version overrides."""
        mmar_dir = Path("mmars")
        model = MODEL_DESC[0]
        model_fullname = f"{model['name']}_{expected_version}"
        with (
            patch.object(mmars, "_get_all_ngc_models") as query,
            patch.object(mmars, "download_and_extract") as download,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            result = download_mmar(model, mmar_dir=mmar_dir, progress=False, api=False, version=version)

        self.assertEqual(result, mmar_dir / model_fullname)
        query.assert_not_called()
        download.assert_called_once_with(
            url=f"https://api.ngc.nvidia.com/v2/models/nvidia/med/{model['name']}/versions/{expected_version}/zip",
            filepath=mmar_dir / f"{model_fullname}.zip",
            output_dir=mmar_dir / model_fullname,
            hash_val=model["hash_val"],
            hash_type=model["hash_type"],
            file_type=model["file_type"],
            has_base=False,
            progress=False,
        )

    @parameterized.expand(
        [
            (name, version, expected_warnings)
            for name in ("get_model_spec", "download_mmar", "load_from_mmar", "RemoteMMARKeys")
            for version, expected_warnings in (
                ("1.6.0", 0),
                ("1.7.0rc1", 0),
                ("1.7.0", 1),
                ("1.7.1", 1),
                ("1.9.0rc1", 1),
                ("1.9.0", 0),
            )
        ]
    )
    def test_deprecation_warning(self, name, version, expected_warnings):
        """Warn from 1.7.0, remove callable APIs in 1.9.0, and preserve MMAR constants."""
        # Import an isolated copy to test the actual decorators across the release
        # boundary without changing the functions imported by other tests.
        source = model_desc if name == "RemoteMMARKeys" else mmars
        spec = spec_from_file_location("monai.apps.mmars._mmars_deprecation_test", source.__file__)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        with patch("monai.utils.deprecate_utils.deprecated", partial(deprecated, version_val=version)):
            spec.loader.exec_module(module)

        if name == "RemoteMMARKeys":
            action = module.RemoteMMARKeys
            with warnings.catch_warnings(record=True) as constant_warnings:
                warnings.simplefilter("always")
                self.assertEqual(module.MODEL_DESC, MODEL_DESC)
                for key, value in vars(model_desc.RemoteMMARKeys).items():
                    if key.isupper():
                        self.assertEqual(getattr(module.RemoteMMARKeys, key), value)
            self.assertEqual(len(constant_warnings), 0)
        elif name == "download_mmar":
            action = partial(module.download_mmar, MODEL_DESC[0], mmar_dir="mmars", api=False)
        elif name == "load_from_mmar":
            action = partial(module.load_from_mmar, "test_model")
        else:
            action = partial(module.get_model_spec, 0)

        with ExitStack() as stack, warnings.catch_warnings(record=True) as captured:
            if name != "RemoteMMARKeys":
                stack.enter_context(patch.object(module, "download_and_extract"))
            warnings.simplefilter("always", FutureWarning)
            if version == "1.9.0":
                with self.assertRaisesRegex(DeprecatedError, rf"{name}.*removed in version 1\.9\.0.*MONAI Bundles"):
                    action()
            elif name == "load_from_mmar":
                with self.assertRaisesRegex(DeprecatedError, "disabled.*security.*MONAI Bundles"):
                    action()
            else:
                action()

        future_warnings = [warning for warning in captured if issubclass(warning.category, FutureWarning)]
        self.assertEqual(len(future_warnings), expected_warnings)
        if expected_warnings:
            self.assertRegex(
                str(future_warnings[0].message),
                rf"{name}.*deprecated since version 1\.7\.0.*removed in version 1\.9\.0.*MONAI Bundles",
            )

        if name == "load_from_mmar":
            with self.assertRaisesRegex(DeprecatedError, "disabled.*security.*MONAI Bundles"):
                inspect.unwrap(module.load_from_mmar)("test_model")

    def test_download_mmar_string_item_warns_once(self):
        """Resolving a string model ID inside `download_mmar` must not add a second deprecation warning."""
        spec = spec_from_file_location("monai.apps.mmars._mmars_deprecation_test", mmars.__file__)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        with patch("monai.utils.deprecate_utils.deprecated", partial(deprecated, version_val="1.7.0")):
            spec.loader.exec_module(module)

        with patch.object(module, "download_and_extract"), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", FutureWarning)
            module.download_mmar(MODEL_DESC[0]["id"], mmar_dir="mmars", api=False)

        future_warnings = [warning for warning in captured if issubclass(warning.category, FutureWarning)]
        self.assertEqual(len(future_warnings), 1)
        self.assertRegex(str(future_warnings[0].message), "download_mmar.*deprecated")

    @parameterized.expand(
        [
            ("default", "test_model", {}),
            ("index", 0, {}),
            ("model_spec", MODEL_DESC[0], {"api": False}),
            ("weights_only", "test_model", {"weights_only": True}),
            ("untrained", "test_model", {"pretrained": False}),
            ("torchscript", "test_model", {"model_file": "models/model.ts"}),
            ("custom_directory", "test_model", {"mmar_dir": Path("mmars"), "map_location": "cpu"}),
        ]
    )
    def test_load_disabled_without_side_effects(self, _, item, kwargs):
        """Reject every loading mode without downloads, file reads, or deserialization."""
        with ExitStack() as stack:
            guards = [
                stack.enter_context(patch(target, side_effect=AssertionError(f"Unexpected call to {target}")))
                for target in (
                    "monai.apps.mmars.mmars.download_mmar",
                    "monai.apps.mmars.mmars.download_and_extract",
                    "monai.apps.mmars.mmars._get_all_ngc_models",
                    "torch.load",
                    "torch.jit.load",
                )
            ]
            guards.append(
                stack.enter_context(
                    patch.object(mmars, "open", create=True, side_effect=AssertionError("Unexpected file access"))
                )
            )
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore", FutureWarning)
            for loader in (load_from_mmar, inspect.unwrap(load_from_mmar)):
                with self.assertRaisesRegex(DeprecatedError, "disabled.*security.*MONAI Bundles"):
                    loader(item, **kwargs)
            for guard in guards:
                guard.assert_not_called()

    def test_public_exports(self):
        """Retain the public function and descriptor-class aliases for existing imports."""
        for name in ("get_model_spec", "download_mmar", "load_from_mmar"):
            self.assertIs(getattr(monai.apps, name), getattr(mmars, name))
            self.assertIs(getattr(monai.apps.mmars, name), getattr(mmars, name))
        self.assertIs(monai.apps.RemoteMMARKeys, model_desc.RemoteMMARKeys)
        self.assertIs(monai.apps.mmars.RemoteMMARKeys, model_desc.RemoteMMARKeys)


if __name__ == "__main__":
    unittest.main()
