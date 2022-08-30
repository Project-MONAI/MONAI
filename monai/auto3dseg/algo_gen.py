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

from monai.transforms import Randomizable


class Algo:
    """
    An algorithm in this context is loosely defined as a data processing pipeline consisting of multiple components
    such as image preprocessing, followed by deep learning model training and evaluation.
    """

    def set_data_stats(self, *args, **kwargs):
        """Provide dataset (and summaries) so that the model creation can depend on the input datasets."""
        pass

    def train(self, params: dict):
        """Read training/validation data and output a model."""
        pass

    def get_score(self, *args, **kwargs):
        """Returns the model quality measurement based on training and validation datasets."""
        pass


class AlgoGen(Randomizable):
    """
    A data-driven algorithm generator optionally takes the following inputs:

        - training dataset properties (such as data statistics from ``monai.auto3dseg.analyzer``),
        - previous algorithm's scores measuring the model's quality,
        - computational budgets.

    It generates ``Algo`` to be trained with the training datasets.

    ..code-block::

                                  scores
                        +------------------------+
                        |   +---------+          |
        +-----------+   +-->|         |    +-----+----+
        | Dataset,  |       | AlgoGen |--->|   Algo   |
        | summaries |------>|         |    +----------+
        +-----+-----+       +---------+          ^
              |                                  |
              +----------------------------------+

    It maintains a history of previously generated Algo, and their corresponding validation scores.
    The Algo generation process may be stochastic (using ``Randomizable.R`` as the source random state).
    """

    def set_data_stats(self, *args, **kwargs):
        """Provide dataset summaries/properties so that the generator can be conditioned on the input datasets."""
        pass

    def set_budget(self, *args, **kwargs):
        """Provide computational budget so that the generator outputs algorithms that requires reasonable resources."""
        pass

    def set_score(self, *args, **kwargs):
        """Feedback from the previously generated algo, the score can be used for new Algo generations."""
        pass

    def get_data_stats(self, *args, **kwargs):
        """Get current dataset summaries."""
        pass

    def get_budget(self, *args, **kwargs):
        """Get the current computational budget."""
        pass

    def get_history(self, *args, **kwargs):
        """Get the previously generated algo."""
        pass

    def generate(self):
        """Generate new Algo -- based on data_stats, budget, and history of previous algo generations."""
        pass

    def run_algo(self, *args, **kwargs):
        """
        Run the algorithms. This is useful for light-weight Algos where there's no need to
        distribute the Algo running jobs.

        If the generated Algos require significant scheduling of parallel execution,
        a separate controller is preferred to run them.  In this case the controller should also report back
        the scores and the algo history, so that the future AlgoGen.generate can leverage the information.
        """
        pass
