# Data Directory

This directory stores the datasets used by the project.

## Directory Structure
- `raw/`: Stores the raw IMDb dataset downloaded via Hugging Face `datasets` or cached locally.
- `processed/`: Stores processed partitions (train, validation, test) saved as parquet/csv for fast and reproducible loading.

## Dataset Policy
Raw and processed dataset splits are excluded from Git version control via `.gitignore` to keep repository size lean and prevent data leakage.
