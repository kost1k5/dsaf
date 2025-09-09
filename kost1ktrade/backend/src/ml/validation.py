import numpy as np
import pandas as pd

class PurgedTimeSeriesSplit():
    """
    A cross-validator that implements purged and embargoed splits for time series data,
    following the principles from "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

    It creates sequential splits, where the training set expands and the test set slides forward.
    - Purging: Removes training samples whose labels depend on information inside the test period.
    - Embargo: Adds a gap after each test set to prevent leakage from serial correlation.
    """
    def __init__(self, n_splits=5, purge_buffer_days=5, embargo_pct=0.01):
        """
        :param n_splits: Number of splits.
        :param purge_buffer_days: A buffer (in days) to purge from the training set
                                  before the validation set starts.
        :param embargo_pct: Percentage of the entire dataset to use as an embargo gap
                            between the training and test sets.
        """
        self.n_splits = n_splits
        self.purge_buffer_days = purge_buffer_days
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X: pd.DataFrame, y: pd.Series = None, event_end_times: pd.Series = None):
        """
        Generates purged and embargoed train/test indices.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex.")
        if event_end_times is None:
            raise ValueError("The 'event_end_times' series is required for purging.")

        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.embargo_pct)
        purge_buffer = pd.Timedelta(days=self.purge_buffer_days)

        # Create the boundaries for each fold, similar to TimeSeriesSplit
        # We will have n_splits + 1 segments
        fold_bounds = np.linspace(0, n_samples, self.n_splits + 2, dtype=int)

        for i in range(self.n_splits):
            test_start = fold_bounds[i+1]
            test_end = fold_bounds[i+2]

            # The training set ends before the embargo period begins
            train_end = test_start - embargo_size

            if train_end < 0:
                continue

            # --- Define initial train/test splits ---
            train_indices_initial = indices[0:train_end]
            test_indices = indices[test_start:test_end]

            if len(train_indices_initial) == 0:
                continue

            # --- Apply Purging ---
            test_start_time = X.index[test_start]
            train_label_end_times = event_end_times.iloc[train_indices_initial]

            overlapping_indices = train_label_end_times[train_label_end_times > (test_start_time - purge_buffer)].index

            overlapping_locs = X.index.get_indexer_for(overlapping_indices)

            purged_train_indices = np.setdiff1d(train_indices_initial, overlapping_locs)

            yield purged_train_indices, test_indices
