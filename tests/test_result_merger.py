# filename: test_result_merger.py

import unittest
import os
import json
import tempfile
from result_merger import ResultMerger

class TestResultMerger(unittest.TestCase):
    """Test cases for the ResultMerger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.merger = ResultMerger()
        
        # Create test directories
        os.makedirs("./combined_results", exist_ok=True)
        
        # Create unique test file paths
        self.test_dir = tempfile.mkdtemp()
        self.test_id = id(self)
        self.test_audio_path = os.path.join(self.test_dir, f"test_audio_{self.test_id}.wav")
        
        # Create a test RTTM file
        self.test_rttm_path = os.path.join(self.test_dir, f"test_diar_rttm_{self.test_id}.txt")
        with open(self.test_rttm_path, 'w') as f:
            f.write("SPEAKER test 1 0.0 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n")
            f.write("SPEAKER test 1 1.5 1.0 <NA> <NA> SPEAKER_01 <NA> <NA>\n")
        
        # Create a test WhisperX JSON file
        self.test_whisperx_path = os.path.join(self.test_dir, f"test_transcript_{self.test_id}.json")
        whisperx_data = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world"
                },
                {
                    "start": 1.5,
                    "end": 2.5,
                    "text": "This is a test"
                }
            ]
        }
        with open(self.test_whisperx_path, 'w') as f:
            json.dump(whisperx_data, f)
    
    def test_merge_results(self):
        """Test merging results from NeMo and WhisperX."""
        success, output_file = self.merger.merge_results(
            self.test_audio_path, self.test_rttm_path, self.test_whisperx_path
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(output_file)
        self.assertTrue(os.path.exists(output_file))
        
        # Verify the content of the merged file
        with open(output_file, 'r') as f:
            content = f.read()
        
        self.assertIn("Speaker SPEAKER_00", content)
        self.assertIn("Speaker SPEAKER_01", content)
        self.assertIn("Hello world", content)
        self.assertIn("This is a test", content)
    
    def test_missing_files(self):
        """Test handling of missing input files."""
        success, output_file = self.merger.merge_results(
            self.test_audio_path, "nonexistent_file.txt", self.test_whisperx_path
        )
        
        self.assertFalse(success)
        self.assertIsNone(output_file)
    
    def test_invalid_json(self):
        """Test handling of invalid WhisperX JSON."""
        # Create an invalid JSON file
        invalid_json_path = os.path.join(self.test_dir, f"invalid_{self.test_id}.json")
        with open(invalid_json_path, 'w') as f:
            f.write("This is not valid JSON")
        
        success, output_file = self.merger.merge_results(
            self.test_audio_path, self.test_rttm_path, invalid_json_path
        )
        
        self.assertFalse(success)
        self.assertIsNone(output_file)
