import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the parent directory to the Python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
