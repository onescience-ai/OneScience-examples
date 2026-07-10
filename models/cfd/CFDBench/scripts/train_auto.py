import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import main


if __name__ == "__main__":
    main(required_task_type="auto", entry_name="scripts/train_auto.py")
