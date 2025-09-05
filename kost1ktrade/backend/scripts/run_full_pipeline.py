import subprocess
import sys
import os

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import settings
from src.core.utils import parse_asset_from_symbol

def run_command(command: list):
    """Runs a command using subprocess and checks for errors."""
    print(f"\n>>> Running command: {' '.join(command)}")
    try:
        # We use pipenv run to execute the scripts in the project's virtual environment
        # The command passed to this function should be the python script and its args
        full_command = ["pipenv", "run", "python"] + command
        subprocess.run(full_command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\n---!!! SCRIPT FAILED: {' '.join(command)} !!!---")
        print(f"---!!! Error: {e} !!!---")
        # Re-raise the exception to stop the master script
        raise
    except FileNotFoundError:
        print("\n---!!! ERROR: 'pipenv' command not found. !!!---")
        print("---!!! Please ensure pipenv is installed and in your PATH. !!!---")
        raise

def main():
    """
    Main script to run the entire quantitative pipeline for all configured symbols.
    """
    print("======================================================")
    print("===  STARTING FULL QUANTITATIVE PIPELINE FOR ALL   ===")
    print("======================================================")

    # Note: collect_all_data.py fetches for all symbols at once.
    # It does not need to be in the loop.
    pipeline_step_0 = [
        "scripts/collect_all_data.py",
        "--days", "1095" # Set to 3 years
    ]

    try:
        run_command(pipeline_step_0)
    except Exception as e:
        print("\nFATAL: Data collection step failed. Cannot proceed.")
        sys.exit(1)


    # These scripts need to be run per-asset
    for symbol in settings.SYMBOLS:
        asset = parse_asset_from_symbol(symbol)
        timeframe = settings.TIMEFRAME

        print(f"\n================ PROCESSING ASSET: {asset} ================")

        pipeline_steps = [
            ["scripts/process_features.py", "--asset", asset, "--timeframe", timeframe],
            ["scripts/apply_labels.py", "--asset", asset, "--timeframe", timeframe],
            ["scripts/select_features.py", "--asset", asset, "--timeframe", timeframe],
            ["scripts/run_backtest.py", "--asset", asset, "--timeframe", timeframe],
            ["scripts/evaluate_model.py", "--asset", asset, "--timeframe", timeframe],
            ["scripts/create_production_model.py", "--asset", asset, "--timeframe", timeframe],
        ]

        try:
            for step_command in pipeline_steps:
                run_command(step_command)
            print(f"\n=== SUCCESSFULLY COMPLETED PIPELINE FOR {asset} ===")
        except Exception as e:
            # (A) Do not ask for interactive input. Log the error and continue by default.
            print(f"\n---!!! PIPELINE FAILED FOR ASSET: {asset} !!!---")
            print(f"---!!! Error was: {e} !!!---")
            print(f"---!!! Continuing to the next asset. !!!---")
            continue # Go to the next symbol in the loop

    print("\n======================================================")
    print("===      FULL PIPELINE RUN COMPLETED             ===")
    print("======================================================")


if __name__ == "__main__":
    main()
