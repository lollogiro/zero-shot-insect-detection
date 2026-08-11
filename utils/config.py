"""Shared configuration constants.

Environment variables (loaded from ``.env`` by ``utils/__init__.py``) can
override these at runtime.
"""

# Model cache directory (overridable via the MODEL_CACHE_DIR env var)
DEFAULT_MODEL_CACHE_DIR = "./models-cache"

# Score threshold for the binary insect/background classifier
BINARY_CLASS_THRESHOLD = 0.5

# COCO category id assigned to every detection (binary insect vs background)
INSECT_CATEGORY_ID = 1
