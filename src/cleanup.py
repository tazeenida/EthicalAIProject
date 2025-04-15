# cleanup.py
import os
import shutil

def clean_directories():
    """Remove all generated data directories to start fresh."""
    directories_to_clean = [
        "data/responses",
        "data/results",
        "data/results/figures",
        "data/debiased"
    ]
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            print(f"Removing {directory}...")
            shutil.rmtree(directory)
            # Recreate empty directory
            os.makedirs(directory, exist_ok=True)
    
    print("Cleanup complete. All data directories have been reset.")

if __name__ == "__main__":
    confirm = input("This will delete all generated data. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        clean_directories()
    else:
        print("Operation cancelled.")