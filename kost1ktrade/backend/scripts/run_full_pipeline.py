import subprocess
import sys
import os
import multiprocessing
import threading
from io import StringIO
import shutil

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings
from src.core.utils import parse_asset_from_symbol
from scripts.pipeline_worker import run_pipeline_for_asset

def run_command(command: list, asset: str = "PIPELINE"):
    """
    Runs a command using subprocess and streams its output in real-time
    by reading stdout and stderr in separate threads to prevent deadlocks.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    executable = sys.executable

    # Thread-safe way to capture stderr output
    stderr_capture = StringIO()

    def stream_reader(pipe, prefix, output_capture=None):
        """Reads and prints lines from a stream (pipe)."""
        if not pipe:
            return
        try:
            for line in iter(pipe.readline, ''):
                print(f"[{prefix}] {line}", end='')
                if output_capture:
                    output_capture.write(line)
        finally:
            pipe.close()

    try:
        process = subprocess.Popen(
            [executable] + command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=base_dir,
            bufsize=1,
            universal_newlines=True
        )

        stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, asset))
        # Use a different prefix for stderr to easily distinguish error lines
        stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, f"{asset}-ERR", stderr_capture))

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        process.wait()

        if process.returncode != 0:
            error_message = f"\n---!!! SCRIPT FAILED: {' '.join(command)} !!!---\n"
            error_message += f"---!!! Return Code: {process.returncode} !!!---\n"
            error_message += f"---!!! STDERR: ---\n{stderr_capture.getvalue()}\n"
            return False, error_message

        return True, "Success"

    except FileNotFoundError:
        error_message = f"---!!! SCRIPT FAILED: Executable not found at {executable} !!!---\n"
        return False, error_message
    except Exception as e:
        error_message = f"---!!! UNEXPECTED ERROR in run_command: {e} !!!---\n"
        return False, error_message

from itertools import repeat

def main():
    """
    Main script to run the entire quantitative pipeline.
    It runs data collection serially, then processes all assets in parallel.
    """
    print("======================================================")
    print("===  STARTING FULL QUANTITATIVE PIPELINE           ===")
    print("======================================================")

    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    # --- Directory Reset ---
    print("\n--- Resetting Output Directories ---")
    DIRS_TO_RESET = ['data', 'reports', 'models']

    for dir_name in DIRS_TO_RESET:
        dir_path = os.path.join(BASE_DIR, dir_name)

        if os.path.exists(dir_path):
            print(f"Deleting existing directory: {dir_path}")
            try:
                shutil.rmtree(dir_path)
                print(f"-> Successfully removed {dir_name} directory.")
            except Exception as e:
                print(f"---!!! FAILED to delete {dir_name} directory: {e} !!!---")
                sys.exit(1)

        print(f"Creating new directory: {dir_path}")
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"-> Successfully created {dir_name} directory.")
        except Exception as e:
            print(f"---!!! FAILED to create {dir_name} directory: {e} !!!---")
            sys.exit(1)

    print("--- Output directories reset complete ---")


    # --- Log File Management ---
    print("\n--- Preparing Logs ---")
    FULL_LOG_PATH = os.path.join(BASE_DIR, 'full_log.txt')

    # Clear the main pipeline log
    if os.path.exists(FULL_LOG_PATH):
        open(FULL_LOG_PATH, 'w').close()
        print(f"Cleared main log file: {FULL_LOG_PATH}")

    # Remove previous asset-specific logs to prevent stale logs
    for symbol in settings.SYMBOLS:
        asset = parse_asset_from_symbol(symbol)
        # Remove old worker logs (e.g., log_BTC_1h.txt)
        worker_log_path = os.path.join(BASE_DIR, f'log_{asset}_{settings.TIMEFRAME}.txt')
        if os.path.exists(worker_log_path):
            try:
                os.remove(worker_log_path)
                print(f"Removed stale worker log: {os.path.basename(worker_log_path)}")
            except OSError as e:
                print(f"Error removing file {worker_log_path}: {e}")

        # Remove old indicator logs
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
    # num_processes = min(multiprocessing.cpu_count(), len(symbols_to_process))

    # --- STABILITY FIX ---
    # The user's environment is running out of memory with parallel processing.
    # We will force serial execution by setting the number of processes to 1.
    num_processes = 1
    print(f"STABILITY FIX: Forcing serial execution with {num_processes} process to conserve memory.")
    print(f"Initializing a pool of {num_processes} processes to handle {len(symbols_to_process)} assets.")

    # Prepare arguments for starmap
    args_for_starmap = zip(symbols_to_process, repeat(timeframe_to_use))

    # Using maxtasksperchild=1 makes the pool more robust by ensuring each asset
    # is processed in a fresh worker process. This can prevent hangs caused by
    # resource leaks or corrupted state in long-running workers.
    with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
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
