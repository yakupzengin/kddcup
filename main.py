"""
Main pipeline for KDD Cup 2022 ESCI Challenge
Orchestrates data loading, feature engineering, training, and evaluation
"""

import argparse
import sys
from pathlib import Path
import subprocess
import pandas as pd

# Add src to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from src.config.config import Config

def run_task_1(experiment_name="baseline_v1"):
    """Run Task 1: Query-Product Ranking"""
    print("=" * 60)
    print("RUNNING TASK 1: QUERY-PRODUCT RANKING")
    print("=" * 60)
    
    # Execute the training script
    script_path = current_dir / "scripts" / "train_task1.py"
    result = subprocess.run([sys.executable, str(script_path)], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Task 1 completed successfully!")
        print(result.stdout)
    else:
        print("❌ Task 1 failed!")
        print("Error:", result.stderr)
        return False
    
    return True

def run_task_2(experiment_name="baseline_v1"):
    """Run Task 2: Multi-class Product Classification"""
    print("=" * 60)
    print("RUNNING TASK 2: MULTI-CLASS PRODUCT CLASSIFICATION")
    print("=" * 60)
    
    # TODO: Implement Task 2 training
    print("Task 2 implementation pending...")
    return True

def run_task_3(experiment_name="baseline_v1"):
    """Run Task 3: Product Substitute Identification"""
    print("=" * 60)
    print("RUNNING TASK 3: PRODUCT SUBSTITUTE IDENTIFICATION")
    print("=" * 60)
    
    # TODO: Implement Task 3 training
    print("Task 3 implementation pending...")
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        Config.EXAMPLES_FILE,
        Config.PRODUCTS_FILE,
        Config.SOURCES_FILE
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
        else:
            print(f"✓ Found {file_path.name}")
    
    if missing_files:
        print("\n⚠️  Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease download the data files from the KDD Cup 2022 website")
        print("and place them in the data/raw/ directory.")
        return False
    
    return True

def setup_environment():
    """Set up the project environment"""
    print("Setting up environment...")
    
    # Create necessary directories
    Config.create_directories()
    
    # Check data files
    if not check_data_files():
        return False
    
    print("✓ Environment setup complete!")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="KDD Cup 2022 ESCI Challenge Pipeline")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], 
                       help="Run specific task (1, 2, or 3)")
    parser.add_argument("--all-tasks", action="store_true",
                       help="Run all tasks sequentially")
    parser.add_argument("--experiment", type=str, default="baseline_v1",
                       help="Experiment name for organizing results")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only set up environment and check data files")
    
    args = parser.parse_args()
    
    print("KDD Cup 2022 ESCI Challenge Pipeline")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed. Exiting.")
        return
    
    if args.setup_only:
        print("✓ Setup completed. Ready to run training!")
        return
    
    # Determine which tasks to run
    tasks_to_run = []
    if args.all_tasks:
        tasks_to_run = [1, 2, 3]
    elif args.task:
        tasks_to_run = [args.task]
    else:
        # Default: run Task 1 only
        tasks_to_run = [1]
        print("No task specified. Running Task 1 (Query-Product Ranking) by default.")
        print("Use --task <number> to run specific task or --all-tasks for all tasks.")
    
    # Run selected tasks
    results = {}
    for task_num in tasks_to_run:
        print(f"\nStarting Task {task_num}...")
        
        if task_num == 1:
            success = run_task_1(args.experiment)
        elif task_num == 2:
            success = run_task_2(args.experiment)
        elif task_num == 3:
            success = run_task_3(args.experiment)
        
        results[task_num] = success
        
        if not success:
            print(f"❌ Task {task_num} failed. Stopping pipeline.")
            break
    
    # Summary
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    
    for task_num, success in results.items():
        status = "✓ PASSED" if success else "❌ FAILED"
        task_name = Config.TASKS[task_num]["name"]
        print(f"Task {task_num} ({task_name}): {status}")
    
    if all(results.values()):
        print("\n🎉 All tasks completed successfully!")
    else:
        print("\n⚠️  Some tasks failed. Check the logs above.")

if __name__ == "__main__":
    main()