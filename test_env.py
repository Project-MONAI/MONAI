"""
Manual environment check for versions of PyTorch and MONAI.

This script helps verify the runtime versions of key libraries during manual testing
of writer error messages or environment troubleshooting.

Note:
- This is intended for manual execution only.
- Consider moving to a diagnostics folder or integrating into automated tests.
"""

import torch
import monai

print("Torch version:", torch.__version__)
print("MONAI version:", monai.__version__)
