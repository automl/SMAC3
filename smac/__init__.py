import os
import sys

import pkg_resources
from smac.utils import dependencies


if sys.version_info < (3, 5, 2):
    raise ValueError("SMAC requires Python 3.5.2 or newer.")


__version__ = pkg_resources.get_distribution("smac").version
__MANDATORY_PACKAGES__ = "\n".join(["%s" % r for r in pkg_resources.get_distribution("smac").requires()])
dependencies.verify_packages(__MANDATORY_PACKAGES__)

if os.name != 'posix':
    print(
        'Detected unsupported operating system: %s. Please be aware, that SMAC might not run on this system.' %
        sys.platform
    )
