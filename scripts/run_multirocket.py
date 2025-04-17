import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket, Rocket
from aeon.transformations.collection import PaddingTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import json
from argparse import ArgumentParser
from tqdm import tqdm
import pickle
from torch_geometric.seed import seed_everything

# train parameters
SEED = 2024
K = 10

def evaluate_simple_classifier(X: np.array, y: np.array, k: int = 5, verbose: bool = False) -> tuple[dict[str, list[int]], list[np.array]]:

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)

    folds_accuracy = []
    folds_f1 = []
    all_cm = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index] # n da para ter um np array com dimensoes diferentes :(
        y_train, y_test = y[train_index], y[test_index]

        clf = RidgeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        folds_accuracy.append(acc)
        folds_f1.append(f1)
        all_cm.append(cm)

        if verbose:
            print(f"- Fold {i + 1} full report")
            print(classification_report(y_test, y_pred))
    
    return {'accuracy': folds_accuracy, 'macro_f1': folds_f1}, all_cm

def get_rocket_features(X: list) -> np.ndarray:

    univar_X_processed = []
    print(f"- Number of channels: {len(X[0])}")

    transformer = PaddingTransformer()
    minirocket = MiniRocket(num_kernels=10_000, n_jobs=5, random_state=SEED)

    X_padded = transformer.fit_transform(X)
    multivar_X_processed = minirocket.fit_transform(X_padded)

    return multivar_X_processed



if __name__ == '__main__':

    seed_everything(SEED)

    parser = ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to the input pickle file')
    parser.add_argument('--log', type=str, help='Path to the output log file')

    args = parser.parse_args()

    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    X = data['X']   
    y = data['y']

    # expecsts data['X'] to be a list of np arrays, (n_examples, channels, length)

    multivar_X_processed = get_rocket_features(X)

    print(f'- Evaluating MultiMiniRocket')
    metrics, all_cms = evaluate_simple_classifier(multivar_X_processed, y, k=K)

    print(f"\t- mean accuracy: {np.mean(metrics['accuracy']):.3f} +/- {np.std(metrics['accuracy']):.3f}")
    print(f"\t- mean f1: {np.mean(metrics['macro_f1']):.3f} +/- {np.std(metrics['macro_f1']):.3f}")

    with open(args.log, 'w') as f:
        json.dump(metrics, f, indent=4)





