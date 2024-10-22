from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from aeon.distances import dtw_pairwise_distance, ddtw_pairwise_distance, adtw_pairwise_distance
from joblib import Parallel, delayed

COLUMNS = [
    'userAcceleration.x',
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
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with the data.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to store the distances.')
    parser.add_argument("--distance", type=str, default='dtw', choices=['dtw', 'adtw', 'ddtw'], help='Distance to be used to compute. Must be implemented by aeon.')
    args = parser.parse_args()

    output_folder = Path(args.output)

    print(f'- Computing distances from {args.input}.')
    print(f'- Saving results to {output_folder}.')
    print(f'- Using {args.distance} distance.')

    # Load the data
    data = pd.read_csv(args.input, index_col=0)

    series = []

    for col in tqdm(COLUMNS, desc='Extracting time series'):
        # TODO: da para paralelizar isso
        X = get_series(data, col)

        # Para teste
        # X = X[:2]

        series.append(X)

    # TODO: no futuro trocar para outra variação do dtw
    if args.distance == 'dtw':
        distance_func = dtw_pairwise_distance
    elif args.distance == 'adtw':
        distance_func = adtw_pairwise_distance
    elif args.distance == 'ddtw':
        distance_func = ddtw_pairwise_distance

    # Disparando  e computando as distâncias
    print(f'- Dispatching {len(COLUMNS)} jobs.')
    features_distances = Parallel(n_jobs=len(COLUMNS))(
        delayed(distance_func)(series[i]) for i in range(len(series))
    )   

    # Salvando distancias
    for distances, col in zip(features_distances, COLUMNS):
        np.save(output_folder / f"{col.replace('.', '-')}_distances.npy", distances)