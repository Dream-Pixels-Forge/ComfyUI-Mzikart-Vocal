import os
import sys
import subprocess
import pkg_resources

def check_and_install_dependencies():
    """Check and install required dependencies."""
    required_packages = [
        "numpy",
        "scipy",
        "torchaudio",
        "soundfile",
        "noisereduce"
    ]
    
    print("Checking dependencies...")
    
    # Get installed packages
    installed = {pkg.key for pkg in pkg_resources.working_set}
    
    # Check which packages need to be installed
    missing = [pkg for pkg in required_packages if pkg.lower() not in installed]
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        
        # Install missing packages
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print("Please install the following packages manually:")
            for pkg in missing:
                print(f"  pip install {pkg}")
    else:
        print("All dependencies are already installed.")

if __name__ == "__main__":
    check_and_install_dependencies()
    print("\nPlease restart ComfyUI to apply changes.")