import os
import sys

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("--- Starting Path Debug ---")

try:
    # This is the same logic used in config.py
    # Get the directory of the current script (debug_settings.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the 'backend' directory
    backend_dir = os.path.dirname(script_dir)
    # Construct the full path to the .env file
    dotenv_path = os.path.join(backend_dir, '.env')

    print(f"Current script directory: {script_dir}")
    print(f"Calculated 'backend' directory: {backend_dir}")
    print(f"Calculated .env path: {dotenv_path}")

    # Check if the file exists at the calculated path
    file_exists = os.path.exists(dotenv_path)
    print(f"Does the file exist at that path? -> {file_exists}")

    if file_exists:
        print("Path seems correct. Attempting to read the first line...")
        try:
            with open(dotenv_path, 'r') as f:
                first_line = f.readline()
                print(f"Successfully read first line: {first_line.strip()}")
            print("--- Path Debug Finished Successfully ---")
        except Exception as e:
            print(f"---!!! Error reading the file: {e} !!!---")
    else:
        print("---!!! .env file NOT FOUND at the calculated path. !!!---")

except Exception as e:
    print(f"---!!! An unexpected error occurred during path debug: {e} !!!---")
    import traceback
    traceback.print_exc()
