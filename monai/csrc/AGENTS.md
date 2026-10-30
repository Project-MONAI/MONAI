# C++/CUDA extensions (csrc)

Optional native extensions: resampling (`resample/`), filtering
(`filtering/`), the LLTM cell (`lltm/`), shared helpers (`utils/`), and the
pybind entry point (`ext.cpp`).

## Things that are easy to get wrong

- The extensions are not built by default. `setup.py` builds them only when
  `BUILD_MONAI=1` is set, and CUDA sources only when a CUDA toolkit is
  available or `FORCE_CUDA=1` is set. Most contributors and CI jobs run
  without them.
- Every extension has, or shares, a pure PyTorch path in the Python
  packages, and MONAI must import and pass its quick tests with the
  extensions absent. If you add an accelerated path, the fallback stays.
- C++ style is governed by the repo-root `.clang-format`; format native
  code with `./runtests.sh --clangformat` instead of formatting by hand.
  Note that it applies clang-format in place and modifies the files, it is
  not a read-only check.
- Python and native implementations of the same operation must agree on
  dtype and device behavior; test both paths when you touch one.
