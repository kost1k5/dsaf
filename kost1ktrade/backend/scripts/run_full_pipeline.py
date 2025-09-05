import subprocess
import sys
import os
import multiprocessing

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings
from src.core.utils import parse_asset_from_symbol

def run_command(command: list):
    """Runs a command using subprocess and captures output."""
    # This function is now used by the worker, so it should return status and output.
    try:
        process = subprocess.run(
            ["pipenv", "run", "python"] + command,
            check=True, text=True, capture_output=True
        )
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"---!!! SCRIPT FAILED: {' '.join(command)} !!!---\n"
        error_message += f"---!!! Return Code: {e.returncode} !!!---\n"
        error_message += f"---!!! STDOUT: ---\n{e.stdout}\n"
        error_message += f"---!!! STDERR: ---\n{e.stderr}\n"
        return False, error_message

def run_pipeline_for_asset(symbol: str):
    """
    Worker function to run the full pipeline for a single asset.
    This function will be executed in a separate process.
    """
    asset = parse_asset_from_symbol(symbol)
    timeframe = settings.TIMEFRAME
    print(f"--- Starting pipeline for {asset} ---")

    pipeline_steps = [
        ["scripts/process_features.py", "--asset", asset, "--timeframe", timeframe],
        ["scripts/apply_labels.py", "--asset", asset, "--timeframe", timeframe],
        ["scripts/select_features.py", "--asset", asset, "--timeframe", timeframe],
        ["scripts/run_backtest.py", "--asset", asset, "--timeframe", timeframe],
        ["scripts/evaluate_model.py", "--asset", asset, "--timeframe", timeframe],
        ["scripts/create_production_model.py", "--asset", asset, "--timeframe", timeframe],
    ]

    for step_command in pipeline_steps:
        print(f"[{asset}] Running command: {' '.join(step_command)}")
        success, output = run_command(step_command)
        if not success:
            print(f"---!!! PIPELINE FAILED FOR ASSET: {asset} at step: {' '.join(step_command)} !!!---")
            print(output) # Print the detailed error message
            return f"FAILED: {asset}"

    print(f"=== SUCCESSFULLY COMPLETED PIPELINE FOR {asset} ===")
    return f"SUCCESS: {asset}"

def main():
    """
    Main script to run the entire quantitative pipeline.
    It runs data collection serially, then processes all assets in parallel.
    """
    print("======================================================")
    print("===  STARTING FULL QUANTITATIVE PIPELINE           ===")
    print("======================================================")

    # Step 1: Run data collection for all assets (must be serial)
    print("\n--- Step 1: Running Serial Data Collection ---")
    # This script now handles its own incremental logic.
    success, output = run_command(["scripts/collect_all_data.py"])
    if not success:
        print("\nFATAL: Data collection step failed. Cannot proceed.")
        print(output)
        sys.exit(1)
    print("\n--- Data collection finished ---")

    # Step 2: Process all assets in parallel
    print("\n--- Step 2: Processing All Assets in Parallel ---")
    symbols_to_process = settings.SYMBOLS

    # Use as many processes as there are CPUs, but not more than the number of symbols
    num_processes = min(multiprocessing.cpu_count(), len(symbols_to_process))
    print(f"Using {num_processes} processes to handle {len(symbols_to_process)} assets.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(run_pipeline_for_asset, symbols_to_process)

    print("\n--- Parallel Processing Results ---")
    success_count = 0
    failure_count = 0
    for result in results:
        print(f"- {result}")
        if result.startswith("SUCCESS"):
            success_count += 1
        else:
            failure_count += 1

    print("\n======================================================")
    print(f"===      FULL PIPELINE RUN COMPLETED             ===")
    print(f"===      Successes: {success_count} | Failures: {failure_count}              ===")
    print("======================================================")

if __name__ == "__main__":
    # To ensure multiprocessing works correctly across platforms
    multiprocessing.freeze_support()
    main()
