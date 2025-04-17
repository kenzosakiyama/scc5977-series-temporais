import numpy as np
import pandas as pd
import pickle as pk

from argparse import ArgumentParser

def build_x_y_pairs(df: pd.DataFrame) -> tuple[list, list]:

    # TODO: transformar isso num script e salvar entradas como npy 

    X = []
    y = []

    for label in df['act'].unique():
        for subj_id in df['id'].unique():
            subj_mask = df['id'] == subj_id
            act_mask = df['act'] == label
            filtered_df = df[subj_mask & act_mask].reset_index()

            X.append(
                np.stack(
                    [
                    filtered_df['userAcceleration.x'].values,
                    filtered_df['userAcceleration.y'].values,
                    filtered_df['userAcceleration.z'].values
                    ]
                )
            )
            y.append(label)

    y = np.array(y)

    return X, y

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, help='Path to the output pickle file')
    parser.add_argument('--mult', action='store_true', help='Whether use multi-channel data or not')
    # parser.add_argument('--channel', type=str, choices=['acc', 'gyro'], help='Sensor channel to use  when method == uni')

    args = parser.parse_args()

    # Load the data
    df = pd.read_csv(args.file)

    # Build the X and y pairs
    X, y = build_x_y_pairs(df)

    data = {
        'X': X,
        'y': y
    }

    # Save the data to a pickle file
    with open(args.output, 'wb') as f:
        pk.dump(data, f)

    print(f"Data saved to {args.output}")