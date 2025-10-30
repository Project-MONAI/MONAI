
# What's new in 1.5.1 🎉🎉

This is a minor update for MONAI to address security concerns and improve compatibility with the newest PyTorch release.

With the upgrade support for PyTorch 2.8, MONAI now directly support NVIDIA GeForce RTX 50 series GPUs and other Blackwell-based GPUs!

- Support up to PyTorch 2.8.
- Security fixes to address advisories [GHSA-x6ww-pf9m-m73m](https://github.com/Project-MONAI/MONAI/security/advisories/GHSA-x6ww-pf9m-m73m), [GHSA-6vm5-6jv9-rjpj](https://github.com/Project-MONAI/MONAI/security/advisories/GHSA-6vm5-6jv9-rjpj), and [GHSA-p8cm-mm2v-gwjm](https://github.com/Project-MONAI/MONAI/security/advisories/GHSA-p8cm-mm2v-gwjm),
- Updated version of supported Huggingface Transformers library to address security advisories raised for it.
- Updated Torchvision pretrained network loading to use current arguments.
- Many minor fixes to identified issues, see release notes for details on merged PRs.
