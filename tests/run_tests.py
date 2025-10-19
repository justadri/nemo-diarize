# filename: run_tests.py

import unittest
import sys

# Import test modules
from test_audio_preprocessor import TestAudioPreprocessor
from test_whisperx_processor import TestWhisperXProcessor
from test_nemo_processor import TestNemoProcessor
from test_result_merger import TestResultMerger
from test_main import TestMain
from test_utils import TestUtils

def run_tests():
    """Run all test cases."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestAudioPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestWhisperXProcessor))
    test_suite.addTest(unittest.makeSuite(TestNemoProcessor))
    test_suite.addTest(unittest.makeSuite(TestResultMerger))
    test_suite.addTest(unittest.makeSuite(TestMain))
    test_suite.addTest(unittest.makeSuite(TestUtils))
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Return the result
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
