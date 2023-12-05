# What's new in 1.3 🎉🎉

- Bundle usability enhancements
- Integrating MONAI Generative into MONAI core


## Bundle usability enhancements

Based on the experience of building MONAI model zoo and the feedback from the community,
MONAI 1.3 provides major enhancements in MONAI Bundle usability. These include:
- Pythonic APIs for Bundle trying to strike a balance between code readability and workflow standardization;
- Streamlined Bundle building processes with step-by-step guides to the concepts;
- Various utility functions for fetching and fine-tuning models from [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo);
- Various fixes for Bundle syntax and documentation, improved test coverage across the Bundle module and Model Zoo.

For more details please visit [the Bundle tutorials](https://github.com/Project-MONAI/tutorials/tree/main/bundle) and
[the Model Zoo demos](https://github.com/Project-MONAI/tutorials/tree/main/model_zoo).

## Integrating MONAI Generative into MONAI Core

Main modules developed at [MONAI GenerativeModels](https://github.com/Project-MONAI/GenerativeModels)
are being ported into the core codebase, allowing for consistent maintenance and release of the key components for generative AI.
As a starting point, loss functions and metrics are integrated into this version.
