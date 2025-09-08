import os
import sys

# Ensure project root is importable so top-level packages like 'scripts' resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

