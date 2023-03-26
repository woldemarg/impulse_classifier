from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from lightgbm import early_stopping
import numpy as np
from joblib import effective_n_jobs
from typing import Tuple, Callable, Optional
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
        self.model = None
        self.n_jobs = n_jobs if n_jobs else effective_n_jobs()

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
                     learning_rate: float) -> object:

        class_counts = np.bincount(y_train)

        class_weights = {
            c: (len(y_train) / (len(set(y_train)) * class_counts[c]))
            for c in set(y_train)}

        model = updater(
            **dict(
                learning_rate=learning_rate,
                class_weight=class_weights))

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_metric=self._custom_score,
            eval_set=(X_eval, y_eval),
            callbacks=[early_stopping(25, verbose=0)])

        return model

    def fit(self, X: np.array, y: np.array) -> None:

        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y,
            test_size=self.hold_out_ratio,
            random_state=self.random_state)

        y_train_copy = y_train.copy()

        learning_rates = np.geomspace(self.max_lr, self.min_lr, self.num_iters)

        delta_positives, delta_confidence = (1, 1)

        sample_weights = np.full(len(y_train), 1)

        for learning_rate in learning_rates:

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

                # print(np.sum(y_train_copy), np.sum(sample_weights))

                model = self._train_model(
                    X_train,
                    X_eval,
                    y_train_copy,
                    y_eval,
                    updater=self.upd_estimator,
                    sample_weights=sample_weights,
                    learning_rate=learning_rate)

                self.model = model

    def predict(self, X) -> np.array:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.array:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)
