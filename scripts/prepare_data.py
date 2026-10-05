"""Download the Darcy_241 dataset and write the training, validation and test splits.

    python scripts/prepare_data.py

About 0.8 GB is downloaded from the Google Drive folder of the FNO authors, the same file
the upstream example uses. The splits take about 0.4 GB. Running it again does nothing.
"""

import argparse

from _bootstrap import ROOT
from pino_darcy.config import load_config
from pino_darcy.data import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="official")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    manifest = prepare(load_config(args.config, args.overrides).data, ROOT)
    for split, (start, stop) in manifest["splits"].items():
        print(f"{split}: samples {start} to {stop - 1}")


if __name__ == "__main__":
    main()
