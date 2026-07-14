import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIR))

from models.partial_autoencoder.train import main


if __name__ == "__main__":
    main()
