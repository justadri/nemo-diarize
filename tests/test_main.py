# filename: test_main.py

import unittest
import os
import tempfile
import soundfile as sf
import numpy as np
from unittest.mock import patch, MagicMock
import main

class TestMain(unittest.TestCase):
    """Test cases for the main module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test audio file with unique name
        self.test_dir = tempfile.mkdtemp()
        self.test_filename = f"test_audio_{id(self)}.wav"
        self.test_audio_path = os.path.join(self.test_dir, self.test_filename)
        
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        sf.write(self.test_audio_path, audio, sample_rate)
    
    @patch('main.check_dependencies')
    @patch('main.AudioPreprocessor')
    @patch('main.WhisperXProcessor')
    @patch('main.NemoProcessor')
    @patch('main.ResultMerger')
    @patch('builtins.input')
    def test_main_function(self, mock_input, mock_merger, mock_nemo, mock_whisperx, mock_preprocessor, mock_check):
        """Test the main function with mocked components."""
        # Mock user input
        mock_input.side_effect = [
            self.test_dir,  # audio directory
            "en",           # language
            "standard"      # preprocessing profile
        ]
        
        # Mock preprocessor
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor_instance.get_available_profiles.return_value = {
            "standard": "Standard profile",
            "telephone": "Telephone profile",
            "noisy": "Noisy profile"
        }
        mock_preprocessor_instance.preprocess_audio.return_value = (np.zeros(16000), 16000)
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        # Mock WhisperX processor
        mock_whisperx_instance = MagicMock()
        mock_whisperx_instance.process_audio.return_value = (True, "whisperx_output.json")
        mock_whisperx.return_value = mock_whisperx_instance
        
        # Mock NeMo processor
        mock_nemo_instance = MagicMock()
        mock_nemo_instance.process_audio.return_value = (True, "nemo_output.txt")
        mock_nemo.return_value = mock_nemo_instance
        
        # Mock result merger
        mock_merger_instance = MagicMock()
        mock_merger_instance.merge_results.return_value = (True, "combined_output.txt")
        mock_merger.return_value = mock_merger_instance
        
        # Run the main function
        main.main()
        
        # Verify the function calls
        mock_check.assert_called_once()
        mock_preprocessor.assert_called_once()
        mock_whisperx.assert_called_once()
        mock_nemo.assert_called_once()
        mock_merger.assert_called_once()
        
        mock_preprocessor_instance.get_available_profiles.assert_called_once()
        mock_preprocessor_instance.preprocess_audio.assert_called_once()
        mock_whisperx_instance.process_audio.assert_called_once()
        mock_nemo_instance.process_audio.assert_called_once()
        mock_merger_instance.merge_results.assert_called_once()
    
    @patch('main.check_dependencies')
    def test_failed_dependencies(self, mock_check):
        """Test handling of failed dependencies installation."""
        mock_check.return_value = False
        
        # Run the main function
        main.main()
        
        # Verify the function calls
        mock_check.assert_called_once()
    
    @patch('main.check_dependencies')
    @patch('builtins.input')
    def test_invalid_directory(self, mock_input, mock_check):
        """Test handling of invalid audio directory."""
        # Mock dependencies installation
        # mock_check.return_value = True
        
        # Mock user input
        mock_input.return_value = "nonexistent_directory"
        
        # Run the main function
        main.main()
        
        # Verify the function calls
        mock_check.assert_called_once()
        mock_input.assert_called_once()
