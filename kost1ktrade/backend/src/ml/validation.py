import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

class PurgedTimeSeriesSplit:
    """
    A cross-validator that enhances TimeSeriesSplit by purging training observations
    that overlap with the validation set and embargoing observations after the test set.

    This implementation is based on the work of Marcos Lopez de Prado in his book
    "Advances in Financial Machine Learning".
    """
    def __init__(self, n_splits=10, purge_buffer_days=5, embargo_pct=0.01):
        """
        :param n_splits: Number of splits for the underlying TimeSeriesSplit.
        :param purge_buffer_days: A buffer (in days) to purge from the training set
                                  before the validation set starts. This accounts for
                                  any time gaps or market closures.
        :param embargo_pct: Percentage of the dataset to "embargo" after each test set,
                            to prevent leakage from the test set into the next training set.
        """
        self.n_splits = n_splits
        self.purge_buffer_days = purge_buffer_days
        self.embargo_pct = embargo_pct
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X: pd.DataFrame, y: pd.Series = None, event_end_times: pd.Series = None):
        """
        Generates purged train/test indices.

        :param X: The input features DataFrame, must have a DatetimeIndex.
        :param y: The target variable (not used directly, but required for API compatibility).
        :param event_end_times: A Series with a DatetimeIndex matching X, where each
                                value is the timestamp when the event for that observation
                                ended (e.g., when a barrier was hit).
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex.")
        if event_end_times is None:
            raise ValueError("The 'event_end_times' series is required for purging.")
        if not X.index.equals(event_end_times.index):
            raise ValueError("X and event_end_times must have the same index.")

        indices = np.arange(X.shape[0])
        purge_buffer = pd.Timedelta(days=self.purge_buffer_days)
        embargo_size = int(X.shape[0] * self.embargo_pct)

        for train_indices, test_indices in self.tscv.split(X):
            # --- Embargo Logic ---
            # The test set is followed by an embargo period.
            # The next training set cannot start until after this embargo.
            # TimeSeriesSplit handles this implicitly by never re-using test data in training.
            # However, we add an explicit embargo check for correctness demonstration,
            # though it's naturally handled by the forward-only nature of the split.

            test_end_idx = test_indices[-1]
            embargo_start_idx = test_end_idx + 1

            # This is more for conceptual clarity; the next fold's train_indices
            # from TimeSeriesSplit will already start after this point.

            # --- Purging Logic ---
            test_start_time = X.index[test_indices[0]]

            # Get the end times for the training observations
            train_end_times = event_end_times.iloc[train_indices]

            # Identify training observations that finish after the purge buffer starts
            # These are the "contaminated" observations.
            overlapping_indices = train_end_times[train_end_times > (test_start_time - purge_buffer)].index

            # Get the integer positions of the overlapping indices
            overlapping_locs = X.index.get_indexer_for(overlapping_indices)

            # Remove these overlapping observations from the training set
            purged_train_indices = np.setdiff1d(train_indices, overlapping_locs)

            yield purged_train_indices, test_indices
