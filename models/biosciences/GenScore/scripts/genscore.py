import sys
from pathlib import Path

GENSCORE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GENSCORE_DIR))

from models.inference import main


if __name__ == "__main__":
    main()
