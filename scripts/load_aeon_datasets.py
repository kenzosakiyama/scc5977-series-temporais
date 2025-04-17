from aeon.datasets import load_classification
from argparse import ArgumentParser
import numpy as np
import pickle as pkl
from sklearn.preprocessing import LabelEncoder

# EthanolConcentration
# 

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the dataset to load")
    parser.add_argument("--output_file", type=str, help="Path to the output pickle file")

    args = parser.parse_args()

    # Load the dataset
    X, y = load_classification(args.name, return_metadata=False)

    print(f"- Successfully loaded {args.name} dataset")
    # Print the shape of the data
    print(f"- Number of samples: {X.shape[0]}")
    print(f"- Number of features: {X.shape[1]}")
    print(f"- X shape: {X.shape}")
    print(f"- Number of classes: {len(np.unique(y))}")

    # Encoding labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Save the dataset to a pickle file
    with open(args.output_file, "wb") as f:
        pkl.dump({"X": X, "y": y}, f)

