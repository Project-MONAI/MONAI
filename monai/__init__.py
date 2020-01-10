import os, sys
from .utils.moduleutils import loadSubmodules


__copyright__ = "(c) 2019 MONAI Consortium"
__version__tuple__ = (0, 0, 1)
__version__ = "%i.%i.%i" % (__version__tuple__)

__basedir__ = os.path.dirname(__file__)


loadSubmodules(sys.modules[__name__], False)  # load directory modules only, skip loading individual files
loadSubmodules(sys.modules[__name__], True)  # load all modules, this will trigger all export decorations
