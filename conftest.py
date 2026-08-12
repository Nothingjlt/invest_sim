"""Root conftest.py — ensures the project root is on sys.path.

This lets pytest discover `src.*` and root-level modules (e.g.
``generate_global_data``) without requiring ``PYTHONPATH=.`` on the
command line.
"""

import sys
import pathlib

# Insert project root (parent of this file) as the first entry on sys.path
# so that ``from src.config import ...`` and similar imports resolve correctly.
_project_root = pathlib.Path(__file__).parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
