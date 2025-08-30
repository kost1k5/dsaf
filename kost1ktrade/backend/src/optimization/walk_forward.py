import pandas as pd
from typing import Generator, Tuple

def walk_forward_splitter(
    data: pd.DataFrame,
    in_sample_len: int,
    out_of_sample_len: int,
    step_size: int
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Creates a generator for walk-forward analysis data splits.

    Yields tuples of (in_sample_df, out_of_sample_df).

    :param data: The full historical dataset (DataFrame), must have a monotonic increasing index.
    :param in_sample_len: The length of the training/optimization period.
    :param out_of_sample_len: The length of the forward-testing period.
    :param step_size: How many periods to step forward for the next window.
    """
    if in_sample_len + out_of_sample_len > len(data):
        raise ValueError("Data length is smaller than the combined in-sample and out-of-sample lengths.")

    start_index = 0
    while start_index + in_sample_len + out_of_sample_len <= len(data):
        in_sample_end_index = start_index + in_sample_len
        out_of_sample_end_index = in_sample_end_index + out_of_sample_len

        in_sample_df = data.iloc[start_index:in_sample_end_index]
        out_of_sample_df = data.iloc[in_sample_end_index:out_of_sample_end_index]

        yield in_sample_df, out_of_sample_df

        start_index += step_size

# Example Usage
if __name__ == '__main__':
    # Add project root to path for script execution
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    print("--- Walk-Forward Splitter Demonstration ---")
    # Create a dummy dataframe with 100 periods
    dummy_data = pd.DataFrame({'price': range(100)})

    in_len = 20
    out_len = 5
    step = 5

    print(f"Total data: {len(dummy_data)} periods.")
    print(f"In-sample length: {in_len}, Out-of-sample length: {out_len}, Step size: {step}\n")

    split_count = 0
    for i, (in_sample, out_of_sample) in enumerate(walk_forward_splitter(dummy_data, in_len, out_len, step)):
        print(f"Split {i+1}:")
        print(f"  In-sample period:     index {in_sample.index[0]} to {in_sample.index[-1]}")
        print(f"  Out-of-sample period: index {out_of_sample.index[0]} to {out_of_sample.index[-1]}")
        split_count += 1

    print(f"\nGenerated {split_count} splits.")
