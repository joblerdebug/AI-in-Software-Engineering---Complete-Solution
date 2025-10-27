"""
Task 2: Automated Testing with AI
Author: [Your Name]
Date: [Current Date]
"""

import time
import random
from datetime import datetime


class LoginPageSimulator:
    """Simulates a login page for automated testing"""
    
    def __init__(self):
        self.users = {
            'valid_user': 'correct_password',
            'admin': 'admin123',
            'test_user': 'test_pass',
            'developer': 'dev_pass'
        }
        self.login_attempts = []
        self.locked_accounts = set()
    
    def login(self, username, password):
        """Simulate login attempt with comprehensive logging"""
        result = {
            'username': username,
            'timestamp': datetime.now(),
            'success': False,
            'message': '',
            'status_code': 200
        }
        
        # Check if account is locked
        if username in self.locked_accounts:
            result['message'] = 'Account temporarily locked. Please try again later.'
            result['status_code'] = 423
        # Check credentials
        elif username in self.users and self.users[username] == password:
            result['success'] = True
            result['message'] = 'Login successful! Redirecting to dashboard...'
            result['status_code'] = 200
        else:
            result['message'] = 'Invalid credentials! Please try again.'
            result['status_code'] = 401
            
            # Simulate account locking after 3 failed attempts
            failed_attempts = len([attempt for attempt in self.login_attempts 
                                 if attempt['username'] == username and not attempt['success']])
            if failed_attempts >= 2:  # 3rd attempt will lock
                self.locked_accounts.add(username)
                result['message'] = 'Too many failed attempts. Account locked for 30 minutes.'
                result['status_code'] = 423
        
        self.login_attempts.append(result)
        return result


class TestExecutor:
    """Executes comprehensive test scenarios"""
    
    def __init__(self):
        self.simulator = LoginPageSimulator()
        self.test_results = []
    
    def execute_test_case(self, username, password, expected_success, description):
        """Execute individual test case"""
        result = self.simulator.login(username, password)
        
        test_result = {
            'description': description,
            'username': username,
            'expected_success': expected_success,
            'actual_success': result['success'],
            'message': result['message'],
            'status_code': result['status_code'],
            'passed': result['success'] == expected_success,
            'timestamp': result['timestamp']
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def run_comprehensive_tests(self):
        """Run a comprehensive test suite"""
        test_cases = [
            # Valid scenarios
            ('valid_user', 'correct_password', True, 'Valid credentials'),
            ('admin', 'admin123', True, 'Admin login'),
            ('developer', 'dev_pass', True, 'Developer login'),
            
            # Invalid scenarios
            ('valid_user', 'wrong_password', False, 'Wrong password'),
            ('invalid_user', 'some_password', False, 'Invalid username'),
            ('', 'password', False, 'Empty username'),
            ('test_user', '', False, 'Empty password'),
            ('nonexistent', 'pass', False, 'Non-existent user'),
            
            # Edge cases
            ('VALID_USER', 'correct_password', False, 'Case sensitivity'),
            ('valid_user', 'Correct_Password', False, 'Password case sensitivity'),
            ('admin', 'ADMIN123', False, 'Admin case sensitivity')
        ]
        
        print("🚀 Starting Comprehensive Automated Testing...")
        print("=" * 60)
        
        for i, (username, password, expected_success, description) in enumerate(test_cases, 1):
            # Add small delay to simulate real testing
            time.sleep(0.1)
            
            test_result = self.execute_test_case(username, password, expected_success, description)
            
            status = "✅ PASS" if test_result['passed'] else "❌ FAIL"
            print(f"Test {i:2d}: {description}")
            print(f"       Input: username='{username}', password='{'*' * len(password) if password else ''}'")
            print(f"       Expected: {'Success' if expected_success else 'Failure'}")
            print(f"       Actual: {'Success' if test_result['actual_success'] else 'Failure'}")
            print(f"       Message: {test_result['message']}")
            print(f"       Result: {status}")
            print("-" * 50)
        
        return self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100
        
        report = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'test_results': self.test_results,
            'execution_time': datetime.now()
        }
        
        print(f"\n📊 COMPREHENSIVE TEST REPORT")
        print("=" * 40)
        print(f"Total Tests Executed: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Tests Failed: {failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Test Completion: {datetime.now()}")
        
        if failed_tests > 0:
            print(f"\n🔍 Failed Tests Analysis:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['description']}: {result['message']}")
        
        return report


def run_automated_testing():
    """Main function to run automated testing"""
    executor = TestExecutor()
    report = executor.run_comprehensive_tests()
    return report


if __name__ == "__main__":
    run_automated_testing()
