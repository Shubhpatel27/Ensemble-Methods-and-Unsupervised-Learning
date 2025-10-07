import os
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")
# Function to run a Python script
def run_script(script_name):
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script_name}: {e}")

# Main function
def main():
    # List all files in the current directory
    files = os.listdir()
    
    # Check if each file is a Python script and execute it
    for file in files:
        if file.endswith(".py") and file != "main.py":
            print(f"Executing {file}...")
            run_script(file)
            print(f"{file} execution completed.\n")

if __name__ == "__main__":
    main()
