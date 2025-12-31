"""
Description: Split synergy dataset into train, val and test and save splits as csv

Author: Kusal Debnath
"""

from tdc.multi_pred import DrugSyn
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"

RAW_DATA_PATH = DATA_PATH / "raw"
TRAIN_DATA_PATH = RAW_DATA_PATH / "train_split.csv"
VAL_DATA_PATH = RAW_DATA_PATH / "val_split.csv"
TEST_DATA_PATH = RAW_DATA_PATH / "test_split.csv"

def main():

    # Load drug-synergy data
    data = DrugSyn(name="DrugComb")
    split = data.get_split()
    # print(type(split['train']))

    train_split, val_split, test_split = split['train'], split['valid'], split['test']
    # print(f"Train shape: {train_split.shape}")
    # print(f"Validation shape: {val_split.shape}")
    # print(f"Test shape: {test_split.shape}")

    if not TRAIN_DATA_PATH.exists():
        train_split.to_csv(TRAIN_DATA_PATH, index=False)
    else:
        print(f'Train data already exists in {TRAIN_DATA_PATH}')

    if not VAL_DATA_PATH.exists():
        val_split.to_csv(VAL_DATA_PATH, index=False)
    else:
        print(f'Validation data already exists in {VAL_DATA_PATH}')

    if not TEST_DATA_PATH.exists():
        test_split.to_csv(TEST_DATA_PATH, index=False)
    else:
        print(f'Test data already exists in {TEST_DATA_PATH}')

if __name__ == "__main__":
    main()