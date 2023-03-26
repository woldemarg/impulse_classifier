from typing import Callable, Iterable
import numpy as np
from joblib import Parallel, delayed, parallel_backend, effective_n_jobs

# %%


class ParallelMixin:

    @staticmethod
    def do_parallel(
            func: Callable,
            iterable: Iterable,
            concatenate_result: bool = True,
            **kwargs: dict) -> np.array:
        """
        Applies a function to each element of an iterable in parallel and
        returns the results as a numpy array.

        Args:
        _fun (Callable): The function to apply to each element of the iterable.
        _itr (Iterable): The iterable to apply the function to.
        concatenate_result (bool, optional): If True, concatenates the results
        along the second axis. Default is True.
        **kwargs (dict): Additional keyword arguments to pass to the function.

        Returns:
        np.array: The results of applying the function to the iterable,
        in a numpy array.

        Examples:
        def square(x, **kwargs):
            return x**2
        arr = [1, 2, 3, 4, 5]
        result = do_parallel(square, arr, concatenate_result=False)
        print(result)
        # Output: [1, 4, 9, 16, 25]
        """

        # backend = kwargs.get('backend', 'threading')
        backend = kwargs.get('backend', 'loky')

        with parallel_backend(backend, n_jobs=effective_n_jobs()):
            lst_processed = Parallel()(
                delayed(func)(el, **kwargs)
                for el in iterable)

        if concatenate_result:
            return np.concatenate(
                [arr.reshape(-1, 1) for arr in lst_processed], axis=1)

        return lst_processed
