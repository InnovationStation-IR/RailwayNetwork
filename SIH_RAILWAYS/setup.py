#!/usr/bin/env python3
"""
Setup script for Enhanced Railway Simulation System

This script helps set up the simulation environment and run initial tests.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3.8, 0):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("Error: requirements.txt not found")
        return False
    
    try:
        # Install core requirements first
        core_packages = ["simpy>=4.0.0", "networkx>=3.0"]
        for package in core_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        # Install optional packages
        optional_packages = ["pygame", "pandas", "numpy", "psutil"]
        for package in optional_packages:
            try:
                print(f"Installing {package} (optional)...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to install {package} (optional)")
        
        print("✓ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("\nChecking data files...")
    
    required_files = [os.path.join("data", "processed", "godot_station_graph.json")]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✓ Found {file}")
    
    if missing_files:
        print(f"Warning: Missing data files: {missing_files}")
        print("You may need to provide your own railway graph data")
        return False
    
    return True

def run_tests():
    """Run basic tests"""
    print("\nRunning tests...")
    
    test_file = Path("test_simulation.py")
    if not test_file.exists():
        print("Warning: test_simulation.py not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, "test_simulation.py"], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Tests passed successfully")
            return True
        else:
            print("✗ Tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def create_sample_config():
    """Create sample configuration file"""
    print("\nCreating sample configuration...")
    
    sample_config = {
        "simulation": {
            "episode_duration": 3600.0,
            "initial_trains": 20,
            "snapshot_interval": 1.0,
            "collision_detection": True,
            "enable_metrics": True,
            "seed": 42
        },
        "performance": {
            "enable_spatial_indexing": True,
            "enable_path_caching": True,
            "enable_memory_management": True,
            "max_memory_mb": 1024.0
        },
        "visualization": {
            "screen_width": 1280,
            "screen_height": 720,
            "fps": 60,
            "enable_animations": True
        }
    }
    
    config_file = Path("sample_config.json")
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"✓ Sample configuration created: {config_file}")
    return True

def run_demo():
    """Run a quick demo simulation"""
    print("\nRunning demo simulation...")
    
    demo_script = """
import time
from simulation.enhanced import (
    EnhancedSimulation, SimulationConfig, SimulationMode,
    create_sample_od_pairs
)

# Create demo configuration
config = SimulationConfig(
    episode_duration=60.0,  # 1 minute demo
    initial_trains=5,
    snapshot_interval=5.0,
    collision_detection=True,
    enable_metrics=True,
    seed=42
)

# Create simulation
sim = EnhancedSimulation(
    datafile=os.path.join("data", "processed", "godot_station_graph.json"),
    config=config,
    mode=SimulationMode.HEADLESS
)

if sim.is_running:
    # Generate OD pairs
    od_pairs = create_sample_od_pairs(
        stations=sim.stations,
        num_pairs=config.initial_trains,
        max_depart_time=config.episode_duration / 2
    )
    
    # Spawn trains
    sim.spawn_trains_from_ods(od_pairs)
    
    print("Running demo simulation...")
    start_time = time.time()
    
    # Run simulation
    sim.run_headless_episode(config.episode_duration)
    
    real_time = time.time() - start_time
    
    # Print results
    print(f"\\nDemo completed in {real_time:.2f} real seconds")
    print(f"Simulated time: {sim.current_sim_time:.2f} seconds")
    print(f"Total trains: {len(sim.trains)}")
    print(f"Completed trains: {sum(1 for t in sim.trains if not t.is_visible)}")
    print(f"Network utilization: {sim.get_network_utilization():.1%}")
    
    print("\\n✓ Demo simulation completed successfully!")
else:
    print("✗ Failed to initialize simulation")
"""
    
    demo_file = Path("demo.py")
    with open(demo_file, 'w') as f:
        f.write(demo_script)
    
    try:
        result = subprocess.run([sys.executable, "demo.py"], 
                               capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✓ Demo simulation completed successfully")
            print(result.stdout)
            return True
        else:
            print("✗ Demo simulation failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Demo simulation timed out")
        return False
    except Exception as e:
        print(f"Error running demo: {e}")
        return False
    finally:
        # Clean up demo file
        if demo_file.exists():
            demo_file.unlink()

def main():
    """Main setup function"""
    print("Enhanced Railway Simulation System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        print("Warning: Some requirements failed to install")
    
    # Check data files
    check_data_files()
    
    # Create sample configuration
    create_sample_config()
    
    # Run tests
    if not run_tests():
        print("Warning: Tests failed - system may not work correctly")
    
    # Run demo
    if not run_demo():
        print("Warning: Demo failed - check your configuration")
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Review sample_config.json for configuration options")
    print("2. Run: python simul8_enhanced.py visualize")
    print("3. Run: python simul8_enhanced.py simulate --duration 300")
    print("4. Check documentation.py for more examples")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
