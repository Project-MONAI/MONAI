
# What's new in 1.6.2 🎉🎉

This is primarily a security release with fixes for vulnerabilities identified through security alerts and tools.

## Security Fixes

A number of recent security alerts are addressed in this release: 

* GHSA-f45g-w7qf-g6cv: Do not evaluate imports when reading an algorithm score (#9254)
* GHSA-987m-fcvp-pfxm: Treat BundleGen algos mapping values as literal data (#9253)
* GHSA-95v4-gm26-h6xf: Check nnUNet plans network names against allowed modules (#9237)
* GHSA-8wp8-g6wq-2m6r: Restore algo state values literally in algo_from_json (#9233)
* GHSA-jx6w-46x8-v3r5: Confine BundleAlgo.export_to_disk to the output directory (#9236)
* GHSA-9hf8-cr88-f7wg: Keep argument boundaries when building Auto3DSeg commands (#9234)
* GHSA-vq3w-pqfr-2w2r: Quote arguments in the NNI trial command (#9235)
* GHSA-4724-w94g-2rfp: Deprecate MMAR APIs and disable load_from_mmar (#9179)

Other additions include:

* Add ConvNeXt networks for 1D, 2D, and 3D inputs (#8995)
* Add MarchingCubes surface extraction and factor LPIPS into MONAI perceptual loss (#9107, #9287)
* Add average calibration losses, likelihood ratios, and diagnostic odds ratio metrics (#9092, #9105)
* Add per-component Surface Distance and optional native-resolution deep supervision for DynUNet (#8855, #9127)
* Add ROCm support for AMD Instinct GPUs and system resource logging with MLFlowHandler (#9153, #9052)
