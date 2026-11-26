"""CSI positioning backend package."""

import os

# Newer Windows images no longer ship the WMIC executable that joblib uses to detect physical
# cores. This project deliberately runs sklearn KNN with one worker for deterministic inference;
# telling loky the same limit avoids a noisy physical-core probe on every API startup.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
