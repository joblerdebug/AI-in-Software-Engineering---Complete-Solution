"""
Unit tests for Login Simulator
Author: [Your Name]
Date: [Current Date]
"""

import unittest
import sys
import os

sys.path.append('../src')
from automated_testing import LoginPageSimulator, TestExecutor


class TestLoginSimulator(unittest.TestCase):
    """Test cases for LoginPageSimulator"""
    
    def setUp(self):
        self.simulator = LoginPageSimulator()
    
    def test_valid_login(self):
        """Test successful login with valid credentials"""
        result = self.simulator.login('valid_user', 'correct_password')
        self.assertTrue(result['success'])
        self.assertEqual(result['message'], 'Login successful! Redirecting to dashboard...')
    
    def test_invalid_password(self):
        """Test login with invalid password"""
        result = self.simulator.login('valid_user', 'wrong_password')
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], 'Invalid credentials! Please try again.')
    
    def test_invalid_username(self):
        """Test login with invalid username"""
        result = self.simulator.login('nonexistent_user', 'password')
        self.assertFalse(result['success'])
        self.assertEqual(result['message'], 'Invalid credentials! Please try again.')
    
    def test_account_locking(self):
        """Test account locking after multiple failed attempts"""
        # First two attempts should fail but not lock
        for _ in range(2):
            result = self.simulator.login('valid_user', 'wrong_password')
            self.assertFalse(result['success'])
        
        # Third attempt should lock the account
        result = self.simulator.login('valid_user', 'wrong_password')
        self.assertFalse(result['success'])
        self.assertIn('Account locked', result['message'])
        self.assertIn('valid_user', self.simulator.locked_accounts)


class TestTestExecutor(unittest.TestCase):
    """Test cases for TestExecutor"""
    
    def setUp(self):
        self.executor = TestExecutor()
    
    def test_test_execution(self):
        """Test individual test case execution"""
        test_result = self.executor.execute_test_case(
            'valid_user', 'correct_password', True, 'Valid login test'
        )
        
        self.assertEqual(test_result['description'], 'Valid login test')
        self.assertTrue(test_result['passed'])
    
    def test_comprehensive_test_suite(self):
        """Test comprehensive test suite execution"""
        report = self.executor.run_comprehensive_tests()
        
        self.assertIn('total_tests', report)
        self.assertIn('success_rate', report)
        self.assertGreater(report['total_tests'], 0)
        self.assertLessEqual(report['success_rate'], 100)


if __name__ == '__main__':
    unittest.main()
