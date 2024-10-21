from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm
from aeon.distances import dtw_pairwise_distance

COLUMNS = [
    # 'userAcceleration.x',
    'userAcceleration.y',
    'userAcceleration.z',
]

def get_series(df: pd.DataFrame, column: str) -> np.ndarray:

    X = []

    for label in df['act'].unique():
        for subj_id in df['id'].unique():
            subj_mask = df['id'] == subj_id
            act_mask = df['act'] == label
            filtered_df = df[subj_mask & act_mask].reset_index()

            X.append(filtered_df[column].values)
    
    return X

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    print(f'Computing distances from {args.input}.')

    # Load the data
    data = pd.read_csv(args.input, index_col=0)

    for col in tqdm(COLUMNS, desc='Computing distances'):
        X = get_series(data, col)

        # Compute the distances
        # TODO: no futuro trocar para outra variação do dtw
        distances = dtw_pairwise_distance(X)
        
        # Save the distances
        np.save(f"{col.replace('.', '-')}_distances.npy", distances)