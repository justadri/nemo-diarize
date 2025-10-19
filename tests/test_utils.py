# filename: test_utils.py

import unittest
import os
import numpy as np
import tempfile
from utils import format_timestamp

class TestUtils(unittest.TestCase):
    """Test cases for the utils module."""
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test various timestamp values
        test_cases = [
            (0, "00:00.00"),
            (1.5, "00:01.50"),
            (61.75, "01:01.75"),
            (3600.25, "60:00.25"),
            (3723.125, "62:03.12")  # 1h 2m 3.125s
        ]
        
        for seconds, expected in test_cases:
            formatted = format_timestamp(seconds)
            self.assertEqual(formatted, expected)
