"""
Complete Pipeline Runner - ARABCAB Competition
Executes all training and optimization steps in correct order
"""

import subprocess
import sys
import time
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")

def run_script(script_name, description):
    """Run a Python script and capture output"""
    print_header(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully in {elapsed:.1f}s")
            return True
        else:
            print(f"\n❌ {description} failed!")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ {description} timed out (>300s)")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed with error: {e}")
        return False

def main():
    """Run complete pipeline"""
    print_header("CABLEFLOW-AI COMPLETE PIPELINE")
    print("ARABCAB Competition 2026")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    # Pipeline steps
    steps = [
        ("verify_system.py", "System Verification"),
        ("arabcab.py", "Model 1: Cable Health Prediction"),
        ("market_forecast_ml.py", "Model 2: ML Market Forecasting"),
        ("inventory_optimizer.py", "Inventory Optimization (EOQ)")
    ]
    
    results = []
    
    for script, description in steps:
        success = run_script(script, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠️  Warning: {description} failed, but continuing...")
            print("   (Some steps may not produce outputs)")
    
    # Summary
    print_header("PIPELINE SUMMARY")
    
    total_elapsed = time.time() - overall_start
    
    for description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{description:<50} {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{successful}/{total} steps completed successfully")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    
    if successful == total:
        print_header("✅ PIPELINE COMPLETE - SYSTEM READY!")
        print("""
Next step: Launch the dashboard
    
    streamlit run Dashboard.py
    
Then open browser at: http://localhost:8501

The dashboard includes:
  • Overview with KPIs
  • Cable Health Predictor
  • Demand Analysis
  • ML Market Forecast (5 years)
  • Inventory Optimization (EOQ, ROP, costs)
  • Model Accuracy Metrics
  • Model Explainability
  • Data Explorer

All models trained and ready to use!
        """)
    else:
        print_header("⚠️ PIPELINE COMPLETED WITH WARNINGS")
        print("Some steps failed. Check error messages above.")
        print("\nYou can still:")
        print("  1. Review generated outputs in models/ and outputs/")
        print("  2. Run failed steps individually")
        print("  3. Launch dashboard (may have limited functionality)")
    
    return successful == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
