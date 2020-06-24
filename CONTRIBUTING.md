- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
  * [Submitting pull requests](#submitting-pull-requests)
  * [Coding style](#coding-style)
  * [Automatic code formatting](#automatic-code-formatting)
  * [Utility functions](#utility-functions)
  * [Building the documentation](#building-the-documentation)
- [Unit testing](#unit-testing)
- [Linting and code style testing](#linting-and-code-style-testing)
- [The code reviewing process (for the maintainers)](#the-code-reviewing-process)
  * [Reviewing pull requests](#reviewing-pull-requests)
- [Admin tasks](#admin-tasks)
  * [Releasing a new version](#release-a-new-version)

## Introduction


This documentation is intended for individuals and institutions interested in contributing to MONAI. MONAI is an open-source project and, as such, its success relies on its community of contributors willing to keep improving it. Your contribution will be a valued addition to the code base; we simply ask that you read this page and understand our contribution process, whether you are a seasoned open-source contributor or whether you are a first-time contributor.

### Communicate with us

We are happy to talk with you about your needs for MONAI and your ideas for contributing to the project. One way to do this is to create an issue discussing your thoughts. It might be that a very similar feature is under development or already exists, so an issue is a great starting point.

### Does it belong in PyTorch / Ignite  instead of MONAI?

MONAI is based on the Ignite and PyTorch frameworks. These frameworks implement what we consider to be best practice for general deep learning functionality. MONAI builds on these frameworks with a strong focus on medical applications. As such, it is a good idea to consider whether your functionality is medical-application specific or not. General deep learning functionality may be better off in PyTorch; you can find their contribution guidelines [here](https://pytorch.org/docs/stable/community/contribution_guide.html).

## The contribution process

_Pull request early_

We encourage you to create pull requests early. It helps us track the contributions under development, whether they are ready to be merged or not. Change your pull request's title to begin with `[WIP]` until it is ready for formal review.


### Submitting pull requests
All code changes to the master branch must be done via [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).
1. Create a new ticket or take a known ticket from [the issue list][monai issue list].
1. Check if there's already a branch dedicated to the task.
1. If the task has not been taken, [create a new branch in your fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)
of the codebase named `[ticket_id]-[task_name]`.
For example, branch name `19-ci-pipeline-setup` corresponds to [issue #19](https://github.com/Project-MONAI/MONAI/issues/19).
Ideally, the new branch should be based on the latest `master` branch.
1. Make changes to the branch ([use detailed commit messages if possible](https://chris.beams.io/posts/git-commit/)). If the changes introduce new features, make sure that you write [unit tests](#unit-testing).
1. [Create a new pull request](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) from the task branch to the master branch, with detailed descriptions of the purpose of this pull request.
1. Check [the CI/CD status of the pull request][github ci], make sure all CI/CD tests passed.
1. Wait for reviews; if there are reviews, make point-to-point responses, make further code changes if needed.
1. If there're conflicts between the pull request branch and the master branch, pull the changes from the master and resolve the conflicts locally.
1. Reviewer and contributor may have discussions back and forth until all comments addressed.
1. Wait for the pull request to be merged.

### Coding style
Coding style is checked by flake8, using [a flake8 configuration](./setup.cfg) similar to [PyTorch's](https://github.com/pytorch/pytorch/blob/master/.flake8).
For string definition, [f-string](https://www.python.org/dev/peps/pep-0498/) is recommended to use over `%-print` and `format-print` from python 3.6. So please try to use `f-string` if you need to define any string object.
Python code file formatting could be done locally before submitting a pull request (e.g. using [`psf/Black`](https://github.com/psf/black)), or during the pull request review using MONAI's automatic [code formatting workflow](#automatic-code-formatting).

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

### Automatic code formatting
MONAI provides support of automatic Python code formatting via [a customised GitHub action](https://github.com/Project-MONAI/monai-code-formatter).
This makes the project's Python coding style consistent and reduces maintenance burdens.
Commenting a pull request with `/black` triggers the formatting action based on [`psf/Black`](https://github.com/psf/black) (this is implemented with [`slash command dispatch`](https://github.com/marketplace/actions/slash-command-dispatch)).


Steps for the formatting process:
- After submitting a pull request or push to an existing pull request,
make a comment to the pull request to trigger the formatting action.
The first line of the comment must be `/black` so that it will be interpreted by [the comment parser](https://github.com/marketplace/actions/slash-command-dispatch#how-are-comments-parsed-for-slash-commands).
- [Auto] The GitHub action tries to format all Python files (using [`psf/Black`](https://github.com/psf/black)) in the branch and makes a commit under the name "MONAI bot" if there's code change. The actual formatting action is deployed at [project-monai/monai-code-formatter](https://github.com/Project-MONAI/monai-code-formatter).
- [Auto] After the formatting commit, the GitHub action adds an emoji to the comment that triggered the process.
- Repeat the above steps if necessary.

### Utility functions
MONAI provides a set of generic utility functions and frequently used routines.
These are located in [``monai/utils``](./monai/utils/) and in the module folders such as [``networks/utils.py``](./monai/networks/).
Users are encouraged to use these common routines to improve code readability and reduce the code maintenance burdens.

Notably,
- ``monai.module.export`` decorator can make the module name shorter when importing,
for example, ``import monai.transforms.Spacing`` is the equivalent of ``monai.transforms.spatial.array.Spacing`` if
``class Spacing`` defined in file `monai/transforms/spatial/array.py` is decorated with ``@export("monai.transforms")``.

### Building the documentation
To build documentation via Sphinx in`docs/` folder:
```bash
# install the doc-related dependencies
pip install --upgrade pip
pip install -r docs/requirements.txt

# build the docs
cd docs/
make html
```
The above commands build html documentation. Type `make help` for all supported
formats, type `make clean` to remove the current build files.  If there are any
auto-generated files, please run `make clean` command to clean up, before
submitting a pull request.

When new classes or methods are added, it is recommended to:
- build html documentation locally,
- check the auto-generated documentation from python docstrings,
- edit relevant `.rst` files in [`docs/source`](./docs/source) accordingly.

## Unit testing
MONAI tests are located under `tests/`.

- The unit test's file name follows `test_[module_name].py`.
- The integration test's file name follows `test_integration_[workflow_name].py`.

A bash script (`runtests.sh`) is provided to run all tests locally
Please run ``./runtests.sh -h`` to see all options.

To run a particular test, for example `tests/test_dice_loss.py`:
```
python -m tests.test_dice_loss
```

## Linting and code style testing

```bash
# Install the various static analysis tools for development
pip install -r requirements-dev.txt
```

Before submitting a pull request, we recommend that all linting and unit tests
should pass, by running the following command locally:
```
./runtests.sh --codeformat --coverage
```

_If it's not tested, it's broken_

All new functionality should be accompanied by an appropriate set of tests.
MONAI functionality has plenty of unit tests from which you can draw inspiration,
and you can reach out to us if you are unsure of how to proceed with testing.

MONAI's code coverage report is available at [CodeCov](https://codecov.io/gh/Project-MONAI/MONAI).

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


## Admin tasks

### Release a new version
- Prepare [a release note](https://github.com/Project-MONAI/MONAI/releases).
- Checkout a new branch `releases/[version number]` locally from the master branch and push to the codebase.
- Create a tag, for example `git tag -a 0.1a -m "version 0.1a"`.
- Push the tag to the codebase, for example `git push origin 0.1a`.
  This step will trigger package building and testing.
  The resultant packages are automatically uploaded to
  [TestPyPI](https://test.pypi.org/project/monai/).  The packages are also available for downloading as
  repository's artifacts (e.g. the file at https://github.com/Project-MONAI/MONAI/actions/runs/66570977).
- Check the release test at [TestPyPI](https://test.pypi.org/project/monai/), download the artifacts when the CI finishes.
- Upload the packages to [PyPI](https://pypi.org/project/monai/).
  This could be done manually by ``twine upload dist/*``, given the artifacts are unzipped to the folder ``dist/``.
- Publish the release note.

Note that the release should be tagged with a [PEP440](https://www.python.org/dev/peps/pep-0440/) compliant
[semantic versioning](https://semver.org/spec/v2.0.0.html) number.

If any error occurs during the release process, first checkout a new branch from the master, make PRs to the master
to fix the bugs via the regular contribution procedure.
Then rollback the release branch and tag:
 - remove any artifacts (website UI) and tag (`git tag -d` and `git push origin -d`).
 - reset the `releases/[version number]` branch to the latest master:
 ```bash
git checkout master
git pull origin master
git checkout releases/[version number]
git reset --hard master
```
Finally, repeat the tagging and TestPyPI uploading process.
