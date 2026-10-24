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

import unittest
from itertools import product
from typing import Any

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import (
    ConfusionMatrixMetric,
    compute_confusion_matrix_metric,
    do_metric_reduction,
    get_confusion_matrix,
)
from tests.test_utils import assert_allclose

_device = "cuda:0" if torch.cuda.is_available() else "cpu"
# input data
data: dict[Any, Any] = {
    "y_pred": torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]],
        ],
        device=_device,
    ),
    "y": torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
        ],
        device=_device,
    ),
}

data_nan: dict[Any, Any] = {
    # confusion matrix:[[[0,1,2,1],[1,1,1,1],[0,1,2,1]],
    #                   [[0,0,0,4],[0,0,4,0],[0,4,0,0]],
    #                   [[0,0,2,2],[0,0,2,2],[0,4,0,0]]]
    "y_pred": torch.tensor(
        [
            [[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
        ]
    ),
    "y": torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
            [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ]
    ),
}

data_clf: dict[Any, Any] = {
    "y_pred": torch.tensor([[1, 0, 0], [0, 0, 1]]),
    "y": torch.tensor([[1, 0, 0], [0, 1, 0]]),
    "compute_sample": False,
    "include_background": True,
    "metric_name": "tpr",
    "reduction": "mean_channel",
    "get_not_nans": True,
}

# 1. test confusion matrix
TEST_CASE_CONFUSION_MATRIX = [
    data.copy(),
    torch.tensor(
        [
            [[0.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 1.0]],
            [[1.0, 0.0, 3.0, 0.0], [1.0, 0.0, 2.0, 1.0], [1.0, 1.0, 2.0, 0.0]],
        ]
    ),
]

# 2. test metric with compute_sample
TEST_CASES_COMPUTE_SAMPLE = []
TEST_CASES_COMPUTE_SAMPLE_MULTI_METRICS = []
result_mean_batch = [
    torch.tensor([0.5000, 0.5000, 0.5000]),
    torch.tensor([0.1667, 0.2500, 0.3333]),
    torch.tensor([0.8333, 0.7500, 0.6667]),
    torch.tensor([0.5000, 0.7500, 0.2500]),
    torch.tensor([0.8333, 0.5833, 0.8333]),
    torch.tensor([0.5000, 0.5000, 0.5000]),
    torch.tensor([0.5000, 0.2500, 0.7500]),
    torch.tensor([0.1667, 0.4167, 0.1667]),
    torch.tensor([0.5000, 0.0000, 0.6830]),
    torch.tensor([0.5000, 0.4167, 0.2500]),
    torch.tensor([0.7500, 0.6250, 0.6250]),
    torch.tensor([0.6667, 0.6250, 0.5833]),
    torch.tensor([0.5000, 0.5833, 0.3333]),
    torch.tensor([0.3333, 0.2887, 0.1220]),
    torch.tensor([0.5000, 0.6036, 0.3536]),
    torch.tensor([0.3333, 0.2500, 0.1667]),
    torch.tensor([0.3333, 0.3333, 0.0833]),
    torch.tensor([0.0000, 1.0000, 1.5000]),
    torch.tensor([0.7500, 0.7500, 0.7500]),
    torch.tensor([0.0000, 1.0000, 0.0000]),
]
result_mean = [
    torch.tensor([0.5000]),
    torch.tensor([0.2500]),
    torch.tensor([0.7500]),
    torch.tensor([0.5000]),
    torch.tensor([0.7500]),
    torch.tensor([0.5000]),
    torch.tensor([0.5000]),
    torch.tensor([0.2500]),
    torch.tensor([0.5610]),
    torch.tensor([0.3889]),
    torch.tensor([0.6667]),
    torch.tensor([0.6250]),
    torch.tensor([0.4722]),
    torch.tensor([0.2480]),
    torch.tensor([0.4857]),
    torch.tensor([0.2500]),
    torch.tensor([0.2500]),
    torch.tensor([1.6667]),
    torch.tensor([0.7500]),
    torch.tensor([0.3333]),
]
metric_names = [
    "tpr",
    "fpr",
    "tnr",
    "ppv",
    "npv",
    "fnr",
    "fdr",
    "for",
    "pt",
    "ts",
    "acc",
    "ba",
    "f1",
    "mcc",
    "fm",
    "bm",
    "mk",
    "plr",
    "nlr",
    "dor",
]
result: Any = None
for idx, item in enumerate(metric_names):
    for reduction in ["mean", "mean_batch"]:
        TEST_CASE: list[Any] = [data.copy()]
        TEST_CASE[0]["compute_sample"] = True
        TEST_CASE[0]["include_background"] = True
        TEST_CASE[0]["metric_name"] = item
        TEST_CASE[0]["reduction"] = reduction
        TEST_CASE[0]["get_not_nans"] = True
        if reduction == "mean_batch":
            result = result_mean_batch[idx]
        elif reduction == "mean":
            result = result_mean[idx]
        TEST_CASE.append(result)
        TEST_CASES_COMPUTE_SAMPLE.append(TEST_CASE)

# one input to compute multiple metrics
for reduction in ["mean", "mean_batch"]:
    TEST_CASE_MULTIPLE: list[Any] = [data.copy()]
    TEST_CASE_MULTIPLE[0]["compute_sample"] = True
    TEST_CASE_MULTIPLE[0]["include_background"] = True
    TEST_CASE_MULTIPLE[0]["metric_name"] = metric_names
    TEST_CASE_MULTIPLE[0]["reduction"] = reduction
    TEST_CASE_MULTIPLE[0]["get_not_nans"] = True
    if reduction == "mean_batch":
        result = result_mean_batch
    elif reduction == "mean":
        result = result_mean
    TEST_CASE_MULTIPLE.append(result)
    TEST_CASES_COMPUTE_SAMPLE_MULTI_METRICS.append(TEST_CASE_MULTIPLE)

# 3. test metric with compute_sample, denominator may have zeros
TEST_CASES_COMPUTE_SAMPLE_NAN = []
metric_names = ["tpr", "tnr"]
result_sum = [torch.tensor([0.5000]), torch.tensor([4.8333])]
not_nans_sum = [torch.tensor([6]), torch.tensor([8])]
result_sum_batch = [torch.tensor([0.0000, 0.5000, 0.0000]), torch.tensor([1.6667, 2.5000, 0.6667])]
not_nans_sum_batch = [torch.tensor([3.0, 2.0, 1.0]), torch.tensor([2.0, 3.0, 3.0])]
for idx in range(2):
    for reduction in ["sum", "sum_batch"]:
        TEST_CASE = [data_nan.copy()]
        TEST_CASE[0]["compute_sample"] = True
        TEST_CASE[0]["include_background"] = True
        TEST_CASE[0]["reduction"] = reduction
        TEST_CASE[0]["metric_name"] = metric_names[idx]
        TEST_CASE[0]["get_not_nans"] = True
        if reduction == "sum":
            TEST_CASE.append(result_sum[idx])
            TEST_CASE.append(not_nans_sum[idx])
        elif reduction == "sum_batch":
            TEST_CASE.append(result_sum_batch[idx])
            TEST_CASE.append(not_nans_sum_batch[idx])
        TEST_CASES_COMPUTE_SAMPLE_NAN.append(TEST_CASE)

# 4. test classification task
result_clf = torch.tensor(
    [
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]],
    ]
)

TEST_CASES_CLF = [data_clf.copy(), result_clf]

TEST_CASE_PRECISION = [
    {
        "y_pred": torch.zeros([1, 1, 1024, 1024, 44], device=_device),
        "y": torch.zeros([1, 1, 1024, 1024, 44], device=_device),
    },
    torch.tensor([[[0.0, 0.0, 46137344.0, 0.0]]]),
]

# 5. likelihood ratios and diagnostic odds ratio: hand-computed values, including undefined cases
# sample 0: tp=80, fp=20, tn=70, fn=20 -> tpr=0.8, fpr=2/9 -> LR+=3.6; fnr=0.2, tnr=7/9 -> LR-=0.25714...;
#           DOR=(80*70)/(20*20)=14
# sample 1: tp=10, fp=0,  tn=90, fn=0  -> fpr=0 -> LR+ is NaN; fnr=0 -> LR-=0.0; fp*fn=0 -> DOR is NaN
# sample 2: tp=5,  fp=5,  tn=0,  fn=0  -> tpr=1, fpr=1 -> LR+=1; tnr=0 -> LR- is NaN; fp*fn=0 -> DOR is NaN
# sample 3: tp=0,  fp=10, tn=10, fn=10 -> tpr=0, fpr=0.5 -> LR+=0; fnr=1, tnr=0.5 -> LR-=2; DOR=(0*10)/(10*10)=0
# sample 4: tp=0,  fp=0,  tn=0,  fn=0  -> empty class -> LR+ / LR- / DOR are all NaN
# (row order is [tp, fp, tn, fn], matching compute_confusion_matrix output)
TEST_CASE_LR = [
    torch.tensor(
        [
            [80.0, 20.0, 70.0, 20.0],
            [10.0, 0.0, 90.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [0.0, 10.0, 10.0, 10.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
]

# classification-style input, channel-wise hand-computed values, no undefined cases:
# ch0: tp=2, fp=1, tn=1, fn=1 -> LR+=4/3, LR-=2/3, DOR=2; ch1: tp=1, fp=1, tn=2, fn=1 -> LR+=3/2, LR-=3/4, DOR=2
TEST_CASE_LR_CLF = [
    {
        "y_pred": torch.tensor([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]),
        "y": torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]),
        "include_background": True,
        "metric_name": ["positive likelihood ratio", "negative likelihood ratio", "diagnostic odds ratio"],
        "reduction": "sum_batch",
    },
    [torch.tensor([4.0 / 3.0, 3.0 / 2.0]), torch.tensor([2.0 / 3.0, 3.0 / 4.0]), torch.tensor([2.0, 2.0])],
]

# same input with the background channel excluded: only ch1 values remain
TEST_CASE_LR_CLF_NOBG = [
    {
        "y_pred": torch.tensor([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]),
        "y": torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]]),
        "include_background": False,
        "metric_name": ["lr+", "lr-", "dor"],
        "reduction": "sum_batch",
    },
    [torch.tensor([3.0 / 2.0]), torch.tensor([3.0 / 4.0]), torch.tensor([2.0])],
]

# 6. likelihood ratios / DOR on the multi-class data with undefined entries, exercising all-reduction
# modes. From the per-sample confusion matrices in the data_nan comment above
# ([tp, fp, tn, fn] per channel):
#   plr: [[0, 1, 0], [nan, nan, nan], [nan, nan, nan]]
#   nlr: [[1.5, 1.0, 1.5], [nan, nan, nan], [1.0, 1.0, nan]]
#   dor: [[0, 1, 0], [nan, nan, nan], [nan, nan, nan]]
# so with NaN-ignoring reductions: plr sum=1 (3 not-nans) / sum_batch=[0,1,0] ([1,1,1] not-nans);
# nlr sum=6 (5 not-nans) / sum_batch=[2.5,2.0,1.5] ([2,2,1] not-nans); dor same as plr.
TEST_CASES_LR_NAN = []
_metric_names = ["positive likelihood ratio", "negative likelihood ratio", "diagnostic odds ratio"]
_result_sum = [torch.tensor([1.0]), torch.tensor([6.0]), torch.tensor([1.0])]
_notnans_sum = [torch.tensor([3.0]), torch.tensor([5.0]), torch.tensor([3.0])]
_result_sum_batch = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([2.5, 2.0, 1.5]), torch.tensor([0.0, 1.0, 0.0])]
_notnans_sum_batch = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 1.0]), torch.tensor([1.0, 1.0, 1.0])]
# mean_channel reduces within each sample (over channels): plr/dor keep only sample 0 (1/3),
# nlr keeps sample 0 at 4/3 and sample 2 at 1.0; all-NaN samples reduce to 0 with 0 not-nans.
_result_mean_channel = [
    torch.tensor([1.0 / 3.0, 0.0, 0.0]),
    torch.tensor([4.0 / 3.0, 0.0, 1.0]),
    torch.tensor([1.0 / 3.0, 0.0, 0.0]),
]
_notnans_mean_channel = [torch.tensor([3.0, 0.0, 0.0]), torch.tensor([3.0, 0.0, 2.0]), torch.tensor([3.0, 0.0, 0.0])]
for _idx in range(3):
    for _reduction in ["sum", "sum_batch", "mean_channel"]:
        _test_case = [data_nan.copy()]
        _test_case[0]["compute_sample"] = True
        _test_case[0]["include_background"] = True
        _test_case[0]["metric_name"] = _metric_names[_idx]
        _test_case[0]["reduction"] = _reduction
        _test_case[0]["get_not_nans"] = True
        if _reduction == "sum":
            _test_case.append(_result_sum[_idx])
            _test_case.append(_notnans_sum[_idx])
        elif _reduction == "sum_batch":
            _test_case.append(_result_sum_batch[_idx])
            _test_case.append(_notnans_sum_batch[_idx])
        else:
            _test_case.append(_result_mean_channel[_idx])
            _test_case.append(_notnans_mean_channel[_idx])
        TEST_CASES_LR_NAN.append(_test_case)


class TestConfusionMatrix(unittest.TestCase):
    @parameterized.expand([TEST_CASE_CONFUSION_MATRIX])
    def test_value(self, input_data, expected_value):
        # include or ignore background
        input_data["include_background"] = True
        result = get_confusion_matrix(**input_data)
        assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)
        input_data["include_background"] = False
        result = get_confusion_matrix(**input_data)
        assert_allclose(result, expected_value[:, 1:, :], atol=1e-4, rtol=1e-4)
        np.testing.assert_equal(result.device, input_data["y_pred"].device)

    @parameterized.expand(TEST_CASES_COMPUTE_SAMPLE)
    def test_compute_sample(self, input_data, expected_value):
        params = input_data.copy()
        vals = {}
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        metric(**vals)
        result, _ = metric.aggregate()[0]
        assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)

    @parameterized.expand(TEST_CASES_COMPUTE_SAMPLE_MULTI_METRICS)
    def test_compute_sample_multiple_metrics(self, input_data, expected_values):
        params = input_data.copy()
        vals = {}
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        metric(**vals)
        results = metric.aggregate()
        for idx, item in enumerate(results):
            result = item[0]
            expected_value = expected_values[idx]
            assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)

    @parameterized.expand(TEST_CASES_COMPUTE_SAMPLE_NAN)
    def test_compute_sample_with_nan(self, input_data, expected_value, expected_not_nans):
        params = input_data.copy()
        vals = {}
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        metric(**vals)
        result, not_nans = metric.aggregate()[0]
        assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)
        assert_allclose(not_nans, expected_not_nans, atol=1e-4, rtol=1e-4)

    @parameterized.expand([TEST_CASES_CLF])
    def test_clf_with_nan(self, input_data, expected_value):
        params = input_data.copy()
        vals = {}
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        result = metric(**vals)
        assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)
        result, _ = metric.aggregate(reduction="mean_channel")[0]
        expected_value, _ = do_metric_reduction(expected_value, "mean_channel")
        expected_value = compute_confusion_matrix_metric("tpr", expected_value)
        assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)

    @parameterized.expand([TEST_CASE_PRECISION])
    def test_precision(self, input_data, expected_value):
        # include or ignore background
        result = get_confusion_matrix(**input_data)
        assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)
        np.testing.assert_equal(result.device, input_data["y_pred"].device)

    @parameterized.expand(
        product(TEST_CASE_LR, ["lr+", "plr", "Positive Likelihood Ratio", "POSITIVE_LIKELIHOOD_RATIO"])
    )
    def test_likelihood_ratios_plr(self, confusion_matrix, alias):
        """Check the positive likelihood ratio under every advertised alias.

        Args:
            confusion_matrix: a stacked [5, 4] confusion-matrix tensor with manually specified
                counts, each row in ``[tp, fp, tn, fn]`` order.
            alias: the metric-name spelling to resolve (case/space-insensitive).
        """
        plr = compute_confusion_matrix_metric(alias, confusion_matrix)
        assert_allclose(plr[0], torch.tensor(3.6), atol=1e-4, rtol=1e-4)
        self.assertTrue(torch.isnan(plr[1]))
        assert_allclose(plr[2], torch.tensor(1.0), atol=1e-4, rtol=1e-4)
        assert_allclose(plr[3], torch.tensor(0.0), atol=1e-4, rtol=1e-4)
        self.assertTrue(torch.isnan(plr[4]))

    @parameterized.expand(
        product(TEST_CASE_LR, ["lr-", "nlr", "Negative Likelihood Ratio", "NEGATIVE_LIKELIHOOD_RATIO"])
    )
    def test_likelihood_ratios_nlr(self, confusion_matrix, alias):
        """Check the negative likelihood ratio under every advertised alias.

        Args:
            confusion_matrix: a stacked [5, 4] confusion-matrix tensor with manually specified
                counts, each row in ``[tp, fp, tn, fn]`` order.
            alias: the metric-name spelling to resolve (case/space-insensitive).
        """
        nlr = compute_confusion_matrix_metric(alias, confusion_matrix)
        assert_allclose(nlr[0], torch.tensor(0.2 / (70.0 / 90.0)), atol=1e-4, rtol=1e-4)
        assert_allclose(nlr[1], torch.tensor(0.0), atol=1e-4, rtol=1e-4)
        # sample 2 has tn == 0 (zero specificity), so LR- is undefined -> NaN
        self.assertTrue(torch.isnan(nlr[2]))
        assert_allclose(nlr[3], torch.tensor(2.0), atol=1e-4, rtol=1e-4)
        self.assertTrue(torch.isnan(nlr[4]))

    @parameterized.expand(product(TEST_CASE_LR, ["dor", "Diagnostic Odds Ratio", "DIAGNOSTIC_ODDS_RATIO"]))
    def test_likelihood_ratios_dor(self, confusion_matrix, alias):
        """Check the diagnostic odds ratio under every advertised alias.

        Args:
            confusion_matrix: a stacked [5, 4] confusion-matrix tensor with manually specified
                counts, each row in ``[tp, fp, tn, fn]`` order.
            alias: the metric-name spelling to resolve (case/space-insensitive).
        """
        dor = compute_confusion_matrix_metric(alias, confusion_matrix)
        assert_allclose(dor[0], torch.tensor(14.0), atol=1e-4, rtol=1e-4)
        # samples 1 / 2 have fp * fn == 0, so the odds ratio is undefined -> NaN
        self.assertTrue(torch.isnan(dor[1]))
        self.assertTrue(torch.isnan(dor[2]))
        assert_allclose(dor[3], torch.tensor(0.0), atol=1e-4, rtol=1e-4)
        self.assertTrue(torch.isnan(dor[4]))

    @parameterized.expand([TEST_CASE_LR])
    def test_dor_equals_lr_ratio(self, confusion_matrix):
        """DOR must equal LR+ / LR- wherever all three are defined (algebraic consistency).

        Args:
            confusion_matrix: a stacked [5, 4] confusion-matrix tensor with manually specified
                counts, each row in ``[tp, fp, tn, fn]`` order.
        """
        plr = compute_confusion_matrix_metric("plr", confusion_matrix)
        nlr = compute_confusion_matrix_metric("nlr", confusion_matrix)
        dor = compute_confusion_matrix_metric("dor", confusion_matrix)
        defined = ~torch.isnan(plr) & ~torch.isnan(nlr) & ~torch.isnan(dor) & (nlr != 0)
        self.assertGreater(defined.sum().item(), 0)
        assert_allclose(dor[defined], (plr[defined] / nlr[defined]), atol=1e-4, rtol=1e-4)
        # wherever DOR is NaN at least one of LR+ / LR- must be undefined or zero-denominator too
        self.assertTrue(torch.isnan(dor[~defined]).all())

    @parameterized.expand([TEST_CASE_LR_CLF, TEST_CASE_LR_CLF_NOBG])
    def test_likelihood_ratios_clf(self, input_data, expected_values):
        """Check likelihood ratios / DOR through the ``ConfusionMatrixMetric`` classification API.

        Args:
            input_data: keyword arguments for ``ConfusionMatrixMetric`` plus ``y_pred``/``y``
                to feed the metric.
            expected_values: expected per-channel LR+ / LR- / DOR values after aggregation.
        """
        params = input_data.copy()
        vals = {}
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        metric(**vals)
        results = metric.aggregate()
        # one aggregated channel per requested metric, in the same order as expected_values
        self.assertEqual(len(results), len(expected_values))
        for result, expected_value in zip(results, expected_values):
            assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)

    @parameterized.expand(TEST_CASES_LR_NAN)
    def test_likelihood_reductions_with_nan(self, input_data, expected_value, expected_not_nans):
        """Check LR+ / LR- / DOR under NaN-ignoring reductions with undefined entries present.

        Args:
            input_data: keyword arguments for ``ConfusionMatrixMetric`` plus ``y_pred``/``y``
                to feed the metric (the multi-class ``data_nan`` inputs).
            expected_value: expected reduced metric value.
            expected_not_nans: expected count of defined entries feeding the reduction.
        """
        params = input_data.copy()
        vals = {}
        vals["y_pred"] = params.pop("y_pred")
        vals["y"] = params.pop("y")
        metric = ConfusionMatrixMetric(**params)
        metric(**vals)
        result, not_nans = metric.aggregate()[0]
        assert_allclose(result, expected_value, atol=1e-4, rtol=1e-4)
        assert_allclose(not_nans, expected_not_nans, atol=1e-4, rtol=1e-4)

    def test_unknown_metric_name(self):
        """An unimplemented metric name must raise ``NotImplementedError``."""
        with self.assertRaises(NotImplementedError):
            compute_confusion_matrix_metric("diagnostic likelihood ratio", TEST_CASE_LR[0])


if __name__ == "__main__":
    unittest.main()
