# Apps

Application-level packages built on the core library: `detection` (3D
detection with RetinaNet), `deepedit` and `deepgrow` (interactive
segmentation), `pathology`, `nuclick`, `nnunet`, `generation`,
`reconstruction`, `tcia`, `mmars` (the MONAI model zoo metadata), and
`auto3dseg` (the Auto3DSeg runner; its algorithms live in
`monai/auto3dseg`, see that package's `AGENTS.md`).

## Things that are easy to get wrong

- Sample datasets in `datasets.py` are downloaded from the web at runtime.
  Tests that use them need network access and are excluded from offline
  runs; do not make a unit test depend on a download when a small synthetic
  array covers the same path.
- Apps pin down end-to-end workflows, so they compose core APIs (transforms,
  networks, engines) rather than reimplementing them. If an app needs a new
  option, add it to the core component and use it here.
- Several apps wrap external toolkits or model hubs. Those dependencies stay
  optional and are reached through `monai.utils.optional_import`, so that
  `import monai.apps` works without them.
- Version-sensitive metadata (model zoo entries in `mmars`, TCIA collection
  info) is data, not code: update the JSON or metadata file and its tests,
  not the loader.
