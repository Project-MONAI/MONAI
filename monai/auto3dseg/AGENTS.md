# Auto3DSeg

This package holds the Auto3DSeg algorithms: data analysis (`analyzer.py`),
algorithm generation (`algo_gen.py`), operations on the generated bundles
(`operations.py`), and result summaries (`seg_summarizer.py`). The user-facing
runner, bundle generator, and ensemble builder live one level up in
`monai/apps/auto3dseg` (`AutoRunner`, `BundleGen`, `EnsembleBuilder`); most
changes need to consider both places.

## Things that are easy to get wrong

- Auto3DSeg works by generating bundle folders on disk and then running
  them. Paths inside the generated configs are relative to the generated
  work directory; do not bake in absolute paths from the machine that ran
  the generation.
- The data analyzer's statistics drive algorithm selection and the generated
  configs. Changing an analyzer output key silently changes every generated
  bundle, so update the summarizer and the tests together.
- End-to-end Auto3DSeg runs are slow and usually GPU-based. Keep unit tests
  on tiny synthetic datasets and leave full runs to the integration tests.
- Templates for the generated algorithms live with the runner in
  `monai/apps/auto3dseg`; editing generated output by hand is never the fix,
  edit the template or the generator.
