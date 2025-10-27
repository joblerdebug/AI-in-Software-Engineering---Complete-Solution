"""
Task 1: AI-Powered Code Completion Comparison
Author: [Your Name]
Date: [Current Date]
"""

def sort_dicts_manual(dict_list, key):
    """
    Manual implementation: Sort list of dictionaries by key
    
    Args:
        dict_list (list): List of dictionaries to sort
        key (str): Key to sort by
    
    Returns:
        list: Sorted list of dictionaries
    """
    return sorted(dict_list, key=lambda x: x[key])


def sort_dicts_ai(dict_list, key):
    """
    AI-enhanced implementation: Sort with error handling
    
    Args:
        dict_list (list): List of dictionaries to sort
        key (str): Key to sort by
    
    Returns:
        list: Sorted list of dictionaries
    """
    return sorted(dict_list, key=lambda x: x.get(key, float('inf')))


def benchmark_sorting():
    """Benchmark and compare both implementations"""
    import time
    import random
    
    # Generate test data
    test_data = [
        {'name': f'user_{i}', 'age': random.randint(20, 50), 'score': random.randint(1, 100)}
        for i in range(1000)
    ]
    
    # Test manual implementation
    start_time = time.time()
    manual_result = sort_dicts_manual(test_data.copy(), 'age')
    manual_time = time.time() - start_time
    
    # Test AI implementation
    start_time = time.time()
    ai_result = sort_dicts_ai(test_data.copy(), 'age')
    ai_time = time.time() - start_time
    
    print("🏎️  Performance Benchmark:")
    print(f"Manual implementation: {manual_time:.6f} seconds")
    print(f"AI implementation: {ai_time:.6f} seconds")
    print(f"Performance difference: {(manual_time - ai_time):.6f} seconds")
    
    return manual_result, ai_result, manual_time, ai_time


if __name__ == "__main__":
    # Example usage
    sample_data = [
        {'name': 'John', 'age': 25, 'salary': 50000},
        {'name': 'Alice', 'age': 30, 'salary': 60000},
        {'name': 'Bob', 'age': 20, 'salary': 45000}
    ]
    
    print("Original Data:")
    for item in sample_data:
        print(f"  {item}")
    
    print("\n🔹 Manual Sort by Age:")
    manual_sorted = sort_dicts_manual(sample_data, 'age')
    for item in manual_sorted:
        print(f"  {item}")
    
    print("\n🤖 AI Sort by Salary:")
    ai_sorted = sort_dicts_ai(sample_data, 'salary')
    for item in ai_sorted:
        print(f"  {item}")
    
    # Run benchmarks
    benchmark_sorting()
