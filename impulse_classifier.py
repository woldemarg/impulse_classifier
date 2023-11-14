from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from lightgbm import early_stopping
import numpy as np
from joblib import effective_n_jobs
from typing import Tuple, Callable, Optional, Iterable
from tqdm import tqdm
from automl_mixins import ParallelMixin


# %%

class ImPULSEClassifier(ParallelMixin):
    def __init__(self,
                 estimator: object,
                 min_lr: float,
                 max_lr: float,
                 num_iters: int,
                 hold_out_ratio: float,
                 random_state: int,
                 n_jobs: Optional[int] = None) -> None:
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iters = num_iters
        self.hold_out_ratio = hold_out_ratio
        self.upd_estimator = self._update_estimator(estimator)
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs else effective_n_jobs()
        self.model = None
        self.prior = None

    @staticmethod
    def _update_estimator(estm: object) -> Callable:
        def update_hyperparams(**kwargs):
            # new instance of the estimator
            # https://scikit-learn.org/stable/developers/develop.html
            return estm.__class__(**{**estm.get_params(), **kwargs})
        return update_hyperparams

    @staticmethod
    def _custom_score(y_train: np.array,
                      y_hat: np.array) -> Tuple[str, float, bool]:
        with np.errstate(divide='ignore', invalid='ignore'):
            score = average_precision_score(y_train, np.round(y_hat))
        return ('custom_score', score, True)

    def _train_model(self,
                     X_train: np.ndarray,
                     X_eval: np.ndarray,
                     y_train: np.ndarray,
                     y_eval: np.ndarray,
                     updater: Callable,
                     sample_weights: np.ndarray,
                     learning_rate: float,
                     **kwargs) -> object:

        class_counts = np.bincount(y_train)

        class_weights = {
            c: (len(y_train) / (len(set(y_train)) * class_counts[c]))
            for c in set(y_train)}

        model = updater(
            **dict(
                learning_rate=learning_rate,
                class_weight=class_weights),
            **kwargs)

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_metric=self._custom_score,
            eval_set=(X_eval, y_eval),
            callbacks=[early_stopping(25, verbose=0)])

        return model

    def _iterate(self,
                 learning_rates: Iterable,
                 X_train: np.ndarray,
                 X_eval: np.ndarray,
                 y_train: np.ndarray,
                 y_eval: np.ndarray,
                 **kwargs):

        sample_weights = np.full(len(y_train), 1)

        y_train_copy = y_train.copy()

        delta_positives, delta_confidence = (1, 1)

        for learning_rate in tqdm(learning_rates):

            # print(learning_rate)

            delta_positives, delta_confidence = (1, 1) if not self.model else (
                delta_positives, delta_confidence)

            if self.model:

                sum_positives, sum_confidence = (
                    y_train_copy.sum(), sample_weights.sum())

                chunks = np.array_split(X_train, self.n_jobs)

                preds = np.concatenate(
                    self.do_parallel(
                        self.model.predict_proba,
                        chunks,
                        concatenate_result=False))[:, 1]

                bins = np.percentile(preds, q=np.linspace(0, 100, 11))

                # to include the rightmost value
                bins[-1] += np.finfo(float).eps

                bin_indices = np.digitize(preds, bins)

                ones_idxs = bin_indices >= 9

                zeros_idxs = bin_indices <= 2

                y_train_copy[ones_idxs] = 1

                trues_idx = np.where(y_train == 1)[0]

                sample_weights = np.full(X_train.shape[0], 0.5)

                sample_weights[trues_idx] = 1

                sample_weights[ones_idxs] = preds[ones_idxs]

                sample_weights[zeros_idxs] = 1 - preds[zeros_idxs]

                delta_positives = y_train_copy.sum() - sum_positives

                delta_confidence = sample_weights.sum() - sum_confidence

            if delta_positives > 0 or delta_confidence > 0:

                # print(y_train_copy.sum(), sample_weights.sum())

                try:

                    model = self._train_model(
                        X_train,
                        X_eval,
                        y_train_copy,
                        y_eval,
                        updater=self.upd_estimator,
                        sample_weights=sample_weights,
                        learning_rate=learning_rate,
                        **kwargs)

                except ValueError:

                    self.model = None
                    self.prior = None

                    return False

                prior = np.average(
                    a=np.ma.masked_array(
                        y_train_copy,
                        mask=y_train),
                    weights=np.ma.masked_array(
                        sample_weights,
                        mask=y_train))

                self.model = model
                self.prior = prior

        print(f'Added {y_train_copy.sum() - y_train.sum()} new labels.')

        return True

    def fit(self, X: np.array, y: np.array) -> None:

        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y,
            test_size=self.hold_out_ratio,
            random_state=self.random_state,
            stratify=y)

        learning_rates = np.geomspace(
            self.max_lr,
            self.min_lr,
            self.num_iters)

        fitted = self._iterate(
            learning_rates,
            X_train,
            X_eval,
            y_train,
            y_eval)

        n = 1

        while not fitted:

            if n >= 10:
                raise ValueError(
                    'The model training process failed to converge.')

            print('Adjusting parameters for another attempt.')

            self.max_lr *= np.exp(-0.1)
            self.min_lr *= np.exp(-0.1)

            learning_rates = np.geomspace(
                self.max_lr,
                self.min_lr,
                self.num_iters)

            X_train, X_eval, y_train, y_eval = train_test_split(
                X, y,
                test_size=self.hold_out_ratio,
                random_state=n,
                stratify=y)

            fitted = self._iterate(
                learning_rates,
                X_train,
                X_eval,
                y_train,
                y_eval)

            n += 1

    # def predict(self, X) -> np.array:
    #    if self.model is None:
    #        raise ValueError("Model has not been trained yet.")
    #    return self.model.predict(X)

    # def predict_proba(self, X) -> np.array:
    #    if self.model is None:
    #        raise ValueError("Model has not been trained yet.")
    #    return self.model.predict_proba(X)

    def __getattr__(self, attrname):
        if hasattr(self.model, attrname):
            # If the attribute exists in the base estimator, return it.
            return getattr(self.model, attrname)
        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute '{attrname}'")
