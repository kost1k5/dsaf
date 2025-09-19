import subprocess
import sys
import os
import threading
from io import StringIO


class Tee(object):
    """
    A file-like object that writes to multiple files at once.
    This is used to "tee" output to both stdout and a log file.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except (IOError, ValueError):
                # Handle cases where a stream might be closed or unavailable
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except (IOError, ValueError):
                pass

# Adjust the path to allow imports from the 'src' directory
# This is crucial for the worker process to find the project's modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.utils import parse_asset_from_symbol


def run_command(command: list, asset: str = "PIPELINE"):
    """
    Runs a command using subprocess and streams its output in real-time
    by reading stdout and stderr in separate threads to prevent deadlocks.
    """
    # Note: In a worker process, __file__ refers to this worker file.
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


def run_pipeline_for_asset(symbol: str, timeframe: str):
    """
    Worker function to run the full pipeline for a single asset.
    This function is executed in a separate process.
    It redirects its own stdout/stderr to a dedicated log file AND the console.
    """
    asset = parse_asset_from_symbol(symbol)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file_path = os.path.join(base_dir, f'log_{asset}_{timeframe}.txt')

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    result = f"PENDING: {asset}"
    log_file = None

    try:
        log_file = open(log_file_path, 'w')
        # Redirect stdout and stderr to both the original stream and the log file
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        print(f"--- Starting full pipeline for asset: {asset} on timeframe: {timeframe} ---")
        print(f"--- Log file for this process: {log_file_path} ---")

        pipeline_steps = [
            ("Feature Processing", ["scripts/process_features.py", "--asset", asset, "--timeframe", timeframe]),
            ("Label Application", ["scripts/apply_labels.py", "--asset", asset, "--timeframe", timeframe]),
            ("Model Training", ["scripts/train_model.py", "--symbols", asset, "--timeframe", timeframe]),
            ("Model Evaluation", ["scripts/evaluate_model.py", "--asset", asset, "--timeframe", timeframe]),
            ("Backtesting", ["scripts/run_backtest.py", "--asset", asset, "--timeframe", timeframe]),
            ("Production Model Creation", ["scripts/create_production_model.py", "--asset", asset, "--timeframe", timeframe]),
        ]

        for i, (step_name, step_command) in enumerate(pipeline_steps):
            print(f"\n[{asset}] Running Step {i+1}/{len(pipeline_steps)}: {step_name}...")
            print(f"[{asset}] Running command: python {' '.join(step_command)}")
            success, output = run_command(step_command, asset=asset)
            if not success:
                print(f"---!!! PIPELINE FAILED FOR ASSET: {asset} at step: {step_name} !!!---")
                print(output)
                result = f"FAILED: {asset}"
                return result  # Exit early on failure

        print(f"=== SUCCESSFULLY COMPLETED PIPELINE FOR {asset} ===")
        result = f"SUCCESS: {asset}"

    finally:
        # Ensure that stdout and stderr are always restored
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file:
            log_file.close()
        # The return must happen outside the try/finally block
        return result
