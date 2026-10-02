"""Strict dataset checks; failures raise visible errors."""
from dataset import load_dataset, validate_dataset
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    args = p.parse_args()
    data, metadata = load_dataset(args.dataset)
    result = validate_dataset(data, metadata)
    print("DATASET_CHECK=PASS")
    for key, value in result.items():
        print(f"{key}={value}")
    print(f"collection_status={metadata['status']}")


if __name__ == "__main__":
    main()
