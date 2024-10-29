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

def get_series(df: pd.DataFrame, column: str, multivar: bool = False) -> np.ndarray:

    X = []
    y = []

    for label in df['act'].unique():
        for subj_id in df['id'].unique():
            subj_mask = df['id'] == subj_id
            act_mask = df['act'] == label
            filtered_df = df[subj_mask & act_mask].reset_index()

            X.append(filtered_df[column].values)

            if multivar:
                X.append(
                    np.stack([filtered_df[col].values for col in COLUMNS])
                )

            y.append(label)
    
    return X, y

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with the data.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to store the distances.')
    parser.add_argument("--distance", type=str, default='dtw', choices=['dtw', 'adtw', 'ddtw'], help='Distance to be used to compute. Must be implemented by aeon.')
    parser.add_argument("--multivar", action='store_true', help='Use multivariate time series.')

    args = parser.parse_args()

    output_folder = Path(args.output)

    print(f'- Computing distances from {args.input}.')
    print(f'- Saving results to {output_folder}.')
    print(f'- Using {args.distance} distance.')

    # Load the data
    data = pd.read_csv(args.input, index_col=0)


    # cada elemento é um conjunto X de séries temporais
    series = []
    for col in tqdm(COLUMNS, desc='Extracting time series'):
        # TODO: da para paralelizar isso
        X, _ = get_series(data, col)

        # Para teste
        # X = X[:2]
        series.append(X)

    if args.multivar:
        # combinando variaveis. a primeira dimensão representa o número de canais
        print(f'- Using multivariate time series. Columns: {COLUMNS}')
        multivar_series = []
        for values in zip(*series):
            multivar_series.append(np.stack(values))
        series = multivar_series


    # TODO: no futuro trocar para outra variação do dtw
    if args.distance == 'dtw':
        distance_func = dtw_pairwise_distance
    elif args.distance == 'adtw':
        distance_func = adtw_pairwise_distance
    elif args.distance == 'ddtw':
        distance_func = ddtw_pairwise_distance

    if not args.multivar:
        # Disparando e computando as distâncias
        print(f'- Dispatching {len(COLUMNS)} jobs.')
        features_distances = Parallel(n_jobs=len(COLUMNS))(
            delayed(distance_func)(series[i]) for i in range(len(series))
        )   

        # Salvando distancias
        for distances, col in zip(features_distances, COLUMNS):
            np.save(output_folder / f"{col.replace('.', '-')}_distances.npy", distances)
    else:
        # computa apenas uma vez
        features_distances = distance_func(series)
        np.save(output_folder / f"{args.distance}_distances.npy", features_distances)