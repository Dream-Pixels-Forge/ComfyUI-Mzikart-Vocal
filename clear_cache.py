import os
import shutil
import sys

def clear_pycache():
    """Remove all __pycache__ directories to force module reload."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find and remove all __pycache__ directories
    for root, dirs, files in os.walk(base_dir):
        if "__pycache__" in dirs:
            pycache_dir = os.path.join(root, "__pycache__")
            print(f"Removing {pycache_dir}")
            shutil.rmtree(pycache_dir)
    
    print("All __pycache__ directories removed.")

if __name__ == "__main__":
    clear_pycache()
    print("Please restart ComfyUI to apply changes.")