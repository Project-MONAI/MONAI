- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
  * [Submitting pull requests](#submitting-pull-requests)
  * [Coding style](#coding-style)
- [Unit testing](#unit-testing)
- [The code reviewing process (for the maintainers)](#the-code-reviewing-process)

## Introduction


This documentation is intended for individuals and institutions interested in contributing to MONAI. MONAI is an open-source project and, as such, its success relies on its community of contributors willing to keep improving it. Your contribution will be a valued addition to the code base; we simply ask that you read this page and understand our contribution process, whether you are a seasoned open-source contributor or whether you are a first-time contributor.

### Communicate with us

We are happy to talk with you about your needs for MONAI and your ideas for contributing to the project. One way to do this is to create an issue discussing your thoughts. It might be that a very similar feature is under development or already exists, so an issue is a great starting point.

### Does it belong in PyTorch / Ignite  instead of MONAI?

MONAI is based on the Ignite and PyTorch frameworks. These frameworks implement what we consider to be best practice for general deep learning functionality. MONAI builds on these frameworks with a strong focus on medical applications. As such, it is a good idea to consider whether your functionality is medical-application specific or not. General deep learning functionality may be better off in PyTorch; you can find their contribution guidelines [here](https://pytorch.org/docs/stable/community/contribution_guide.html).

## The contribution process

_Pull request early_

We encourage you to create pull requests early. It helps us track the contributions under development, whether they are ready to be merged or not. Tag your pull request as `[WIP]` until it is ready for formal review.


### Submitting pull requests
All code changes to the master branch must be done via pull requests.
1. Create a new ticket or take a known ticket from [the issue list][monai issue list].
1. Check if there's already a branch dedicated to the task.
1. If the task has not been taken, [create a new branch](https://help.github.com/en/desktop/contributing-to-projects/creating-a-branch-for-your-work) in the codebase or a fork of the codebase named `[ticket_id]-[task_name]`. For example, branch name `19-ci-pipeline-setup` corresponds to [issue #19](https://github.com/Project-MONAI/MONAI/issues/19).
Ideally, the new branch should be based on the latest `master` branch.
1. Make changes to the branch ([use detailed commit messages if possible](https://chris.beams.io/posts/git-commit/)). If the changes introduce new features, make sure that you write [unit tests](#unit-testing).
1. [Create a new pull request](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) from the task branch to the master branch, with detailed descriptions of the purpose of this pull request.
1. Check [the CI/CD status of the pull request][github ci], make sure all CI/CD tests passed.
1. Wait for reviews; if there are reviews, make point-to-point responses, make further code changes if needed.
1. If there're conflicts between the pull request branch and the master branch, pull the changes from the master and resolve the conflicts locally .
1. Reviewer and contributor may have discussions back and forth until all comments addressed.
1. Wait for the pull request to be merged.

### Coding style
Coding style is checked by flake8, using [a flake8 configuration](https://github.com/Project-MONAI/MONAI/blob/master/.flake8) similar to [PyTorch's](https://github.com/pytorch/pytorch/blob/master/.flake8).

License information: all source code files should start with this paragraph:
```
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

### Building the documentation
To build documentation via Sphinx in`docs/` folder:
```bash
cd docs/
make html
```
The above commands build html documentation. Type `make help` for all supported formats,
type `make clean` to remove the current build files.

## Unit testing
MONAI tests are located under `tests/`.

- The unit test's file name follows `test_[module_name].py`.
- The integration test's file name follows `integration_[workflow_name].py`.

A bash script (`runtests.sh`) is provided to run all tests locally
Please run ``./runtests.sh -h`` to see all options.

To run a particular test, for example `tests/test_dice_loss.py`:
```
python -m tests.test_dice_loss
```

Before submitting a pull request, we recommend that all linting and unit tests
should pass, by running the following commands locally:
```
flake8 . --count --statistics
./runtests.sh --coverage
```

_If it's not tested, it's broken_

All new functionality should be accompanied by an appropriate set of tests. MONAI functionality has plenty of unit tests from which you can draw inspiration, and you can reach out to us if you are unsure of how to proceed with testing.



## The code reviewing process


### Reviewing pull requests
All code review comments should be specific, constructive, and actionable.
1. Check [the CI/CD status of the pull request][github ci], make sure all CI/CD tests passed before reviewing (contact the branch owner if needed).
1. Read carefully the descriptions of the pull request and the files changed, write comments if needed.
1. Make in-line comments to specific code segments, [request for changes](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-reviews) if needed.
1. Review any further code changes until all comments addressed by the contributors.
1. Merge the pull request to the master branch.
1. Close the corresponding task ticket on [the issue list][monai issue list].

[github ci]: https://github.com/Project-MONAI/MONAI/actions
[monai issue list]: https://github.com/Project-MONAI/MONAI/issues
