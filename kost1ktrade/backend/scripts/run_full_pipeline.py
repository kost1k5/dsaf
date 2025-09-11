import subprocess
import sys
import os
import multiprocessing

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings
from src.core.utils import parse_asset_from_symbol

def run_command(command: list, asset: str = "PIPELINE"):
    """
    Runs a command using subprocess and streams its output in real-time.
    Prepends the asset name to each line for clarity in parallel execution.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    executable = sys.executable

    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            [executable] + command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=base_dir,
            bufsize=1, # Line-buffered
            universal_newlines=True
        )

        # Stream stdout
        if process.stdout:
            for line in process.stdout:
                # Prepend asset to each line for clarity in parallel logs
                print(f"[{asset}] {line}", end='')

        # Wait for the process to complete and get the exit code
        process.wait()

        # Check for errors after streaming all output
        if process.returncode != 0:
            # Capture any remaining error output
            error_output = process.stderr.read() if process.stderr else ""
            error_message = f"\n---!!! SCRIPT FAILED: {' '.join(command)} !!!---\n"
            error_message += f"---!!! Return Code: {process.returncode} !!!---\n"
            error_message += f"---!!! STDERR: ---\n{error_output}\n"
            return False, error_message

        return True, "Success" # Output is already streamed, so we just need status

    except FileNotFoundError:
        error_message = f"---!!! SCRIPT FAILED: Executable not found at {executable} !!!---\n"
        return False, error_message
    except Exception as e:
        error_message = f"---!!! UNEXPECTED ERROR in run_command: {e} !!!---\n"
        return False, error_message

from itertools import repeat

def run_pipeline_for_asset(symbol: str, timeframe: str):
    """
    Worker function to run the full pipeline for a single asset.
    This function will be executed in a separate process.
    """
    asset = parse_asset_from_symbol(symbol)
    print(f"--- Starting full pipeline for asset: {asset} on timeframe: {timeframe} ---")

    pipeline_steps = [
        ("Feature Processing", ["scripts/process_features.py", "--asset", asset, "--timeframe", timeframe]),
        ("Label Application", ["scripts/apply_labels.py", "--asset", asset, "--timeframe", timeframe]),
        ("Model Training", ["scripts/train_xgboost_model.py", "--asset", asset, "--timeframe", timeframe]),
        ("Model Evaluation", ["scripts/evaluate_model.py", "--asset", asset, "--timeframe", timeframe]),
        ("Walk-Forward Optimization", ["scripts/run_wfo.py", "--asset", asset, "--timeframe", timeframe]),
        ("Backtesting", ["scripts/run_backtest.py", "--asset", asset, "--timeframe", timeframe]),
        ("Production Model Creation", ["scripts/create_production_model.py", "--asset", asset, "--timeframe", timeframe]),
    ]

    for i, (step_name, step_command) in enumerate(pipeline_steps):
        print(f"\n[{asset}] Running Step {i+1}/{len(pipeline_steps)}: {step_name}...")
        print(f"[{asset}] Running command: python {' '.join(step_command)}")
        success, output = run_command(step_command, asset=asset)
        if not success:
            print(f"---!!! PIPELINE FAILED FOR ASSET: {asset} at step: {step_name} !!!---")
            print(output) # Print the detailed error message from run_command
            return f"FAILED: {asset}"
        # No need for a success print here as the output is streamed
        print(f"[{asset}] Finished Step {i+1}/{len(pipeline_steps)}: {step_name}.")


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

    # --- Log File Management ---
    print("\n--- Preparing Logs ---")
    BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
    FULL_LOG_PATH = os.path.join(BASE_DIR, 'full_log.txt')

    # Clear the main pipeline log
    if os.path.exists(FULL_LOG_PATH):
        open(FULL_LOG_PATH, 'w').close()
        print(f"Cleared main log file: {FULL_LOG_PATH}")

    # Remove previous asset-specific indicator logs to prevent stale logs
    for symbol in settings.SYMBOLS:
        asset = parse_asset_from_symbol(symbol)
        indicator_log_path = os.path.join(BASE_DIR, f'indicator_log_{asset}.txt')
        if os.path.exists(indicator_log_path):
            try:
                os.remove(indicator_log_path)
                print(f"Removed stale indicator log for {asset}")
            except OSError as e:
                print(f"Error removing file {indicator_log_path}: {e}")
    print("--- Log preparation complete ---")
    # ---

    # Step 1: Run data collection for all assets (must be serial)
    print("\n--- Step 1: Starting Serial Data Collection ---")
    # This script now handles its own incremental logic.
    # We pass the configured number of days for the initial history pull.
    days_arg = str(settings.DATA_HISTORY_DAYS)
    print(f"Data collection will use an initial history of {days_arg} days.")
    success, output = run_command(["scripts/collect_all_data.py", "--days", days_arg])
    if not success:
        print("\nFATAL: Data collection step failed. Cannot proceed.")
        print(output)
        sys.exit(1)
    print("--- Step 1: Data collection finished successfully ---")

    # Step 2: Process all assets in parallel
    print("\n--- Step 2: Starting Parallel Asset Processing ---")
    symbols_to_process = settings.SYMBOLS
    timeframe_to_use = settings.TIMEFRAME
    print(f"Pipeline will be run on the '{timeframe_to_use}' timeframe for all assets.")

    # Use as many processes as there are CPUs, but not more than the number of symbols
    num_processes = min(multiprocessing.cpu_count(), len(symbols_to_process))
    print(f"Initializing a pool of {num_processes} processes to handle {len(symbols_to_process)} assets.")

    # Prepare arguments for starmap
    args_for_starmap = zip(symbols_to_process, repeat(timeframe_to_use))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(run_pipeline_for_asset, args_for_starmap)

    print("\n--- Step 2: Parallel Processing Finished ---")
    print("\n--- Final Results ---")
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
