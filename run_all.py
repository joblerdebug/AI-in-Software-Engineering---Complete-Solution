#!/usr/bin/env python3
"""
Main execution script for AI in Software Engineering Assignment
Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import time
from datetime import datetime


def create_directories():
    """Create necessary directories"""
    directories = [
        'reports/test_results',
        'reports/model_performance', 
        'data',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def run_task_1():
    """Execute Task 1: Code Completion"""
    print("\n" + "="*60)
    print("🎯 TASK 1: AI-Powered Code Completion")
    print("="*60)
    
    try:
        sys.path.append('src')
        from code_completion import benchmark_sorting
        
        print("Running code completion analysis...")
        manual_result, ai_result, manual_time, ai_time = benchmark_sorting()
        
        # Save results
        with open('reports/task1_results.txt', 'w') as f:
            f.write("Task 1: Code Completion Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Execution Time - Manual: {manual_time:.6f}s\n")
            f.write(f"Execution Time - AI: {ai_time:.6f}s\n")
            f.write(f"Performance Improvement: {(manual_time/ai_time):.2f}x\n")
            f.write(f"Analysis completed: {datetime.now()}\n")
        
        print("✅ Task 1 completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in Task 1: {e}")
        return False


def run_task_2():
    """Execute Task 2: Automated Testing"""
    print("\n" + "="*60)
    print("🎯 TASK 2: Automated Testing")
    print("="*60)
    
    try:
        sys.path.append('src')
        from automated_testing import run_automated_testing
        
        print("Running automated testing suite...")
        test_report = run_automated_testing()
        
        # Save detailed report
        import json
        with open('reports/test_results/detailed_report.json', 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        print("✅ Task 2 completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in Task 2: {e}")
        return False


def run_task_3():
    """Execute Task 3: Predictive Analytics"""
    print("\n" + "="*60)
    print("🎯 TASK 3: Predictive Analytics")
    print("="*60)
    
    try:
        sys.path.append('src')
        from predictive_analytics import PredictiveModel
        
        print("Running predictive analytics pipeline...")
        predictor = PredictiveModel()
        metrics = predictor.run_complete_analysis()
        
        # Save metrics
        import json
        with open('reports/model_performance/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("✅ Task 3 completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in Task 3: {e}")
        return False


def generate_final_report(start_time, end_time, task_results):
    """Generate comprehensive final report"""
    print("\n" + "="*60)
    print("📊 GENERATING FINAL REPORT")
    print("="*60)
    
    total_time = end_time - start_time
    successful_tasks = sum(task_results)
    
    report = f"""
COMPREHENSIVE ASSIGNMENT REPORT
{'=' * 40}

EXECUTION SUMMARY:
• Start Time: {start_time}
• End Time: {end_time}
• Total Duration: {total_time:.2f} seconds
• Tasks Completed: {successful_tasks}/3

DETAILED RESULTS:
• Task 1 (Code Completion): {'✅ SUCCESS' if task_results[0] else '❌ FAILED'}
• Task 2 (Automated Testing): {'✅ SUCCESS' if task_results[1] else '❌ FAILED'} 
• Task 3 (Predictive Analytics): {'✅ SUCCESS' if task_results[2] else '❌ FAILED'}

FILES GENERATED:
• Source Code: src/ directory
• Test Reports: reports/test_results/
• Model Performance: reports/model_performance/
• Documentation: docs/ directory

NEXT STEPS:
1. Review generated reports and visualizations
2. Check model performance metrics
3. Verify test results
4. Submit according to assignment guidelines

{'🎉 ALL TASKS COMPLETED! 🎉' if successful_tasks == 3 else '⚠️ Some tasks encountered issues'}
"""
    
    print(report)
    
    # Save report to file
    with open('reports/final_assignment_report.txt', 'w') as f:
        f.write(report)
    
    print("📄 Final report saved to: reports/final_assignment_report.txt")


def main():
    """Main execution function"""
    print("🚀 AI in Software Engineering - Week 4 Assignment")
    print("📅 Started at:", datetime.now())
    print("🔧 Initializing...")
    
    # Create directory structure
    create_directories()
    
    # Record start time
    start_time = time.time()
    
    # Execute all tasks
    task_results = []
    task_results.append(run_task_1())
    task_results.append(run_task_2()) 
    task_results.append(run_task_3())
    
    # Record end time and generate report
    end_time = time.time()
    generate_final_report(start_time, end_time, task_results)


if __name__ == "__main__":
    main()
