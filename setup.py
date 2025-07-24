"""
Setup script for ESCI Challenge project
Run this script to set up the project environment and verify data files
"""
import os
import sys
from pathlib import Path
import subprocess
from src.config.config import Config

def create_init_files():
    """Create __init__.py files for Python packages"""
    init_dirs = [
        "src",
        "src/config", 
        "src/data",
        "src/features",
        "src/models", 
        "src/evaluation",
        "src/utils"
    ]
    
    for dir_path in init_dirs:
        init_file = Path(dir_path) / "__init__.py"
        init_file.parent.mkdir(parents=True, exist_ok=True)
        if not init_file.exists():
            init_file.touch()
            print(f"Created {init_file}")

def check_data_files():
    """Check if required data files exist"""
    data_dir = Path("data/raw")
    required_files = [
        "shopping_queries_dataset_examples.parquet",
        "shopping_queries_dataset_products.parquet", 
        "shopping_queries_dataset_sources.csv"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            missing_files.append(str(file_path))
        else:
            print(f"✓ Found {file_path}")
    
    if missing_files:
        print("\n⚠️  Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease download the data files from the KDD Cup 2022 website")
        print("and place them in the data/raw/ directory.")
        return False
    
    print("\n✓ All required data files found!")
    return True

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def setup_project():
    """Main setup function"""
    print("Setting up ESCI Challenge project...")
    print("="*50)
    
    # Create project structure
    Config.create_directories()
    
    # Create __init__.py files
    create_init_files()
    
    # Check data files
    # data_ok = check_data_files()
    
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    # if data_ok:
    #     print("✓ Project structure created")
    #     print("✓ Data files verified")
    #     print("\nYou can now run the training pipeline:")
    #     print("  python main.py --task 1 --experiment baseline_v1")
    #     print("  python main.py --all-tasks")
    # else:
    #     print("⚠️  Setup incomplete - missing data files")
    #     print("Please download and place the data files before running the pipeline")
    
    # return data_ok

if __name__ == "__main__":
    # Add current directory to Python path
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    setup_project()