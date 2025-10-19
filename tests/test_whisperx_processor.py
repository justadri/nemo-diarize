
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: fixed_test_whisperx_processor.py
# execution: true

import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
from unittest.mock import patch, MagicMock
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the processor to test
from whisperx_processor import WhisperXProcessor

class TestWhisperXProcessor(unittest.TestCase):
    """Test cases for the WhisperXProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = WhisperXProcessor()
        
        # Create a unique test directory name but don't create the directory
        self.test_dir = f"test_dir_{id(self)}"
        self.test_filename = f"test_audio_{id(self)}.wav"
        self.test_audio_path = os.path.join(self.test_dir, self.test_filename)
        
        # Use in-memory audio data instead of writing to disk
        self.sample_rate = 16000
        self.duration = 3  # seconds
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.audio_data = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        # Create output directory
        os.makedirs("./transcription_results", exist_ok=True)
    
    @patch('whisperx_processor.whisperx')
    @patch('whisperx_processor.DiarizationPipeline')
    def test_process_audio_with_mocks(self, mock_diarize_pipeline_class, mock_whisperx):
        """Test processing audio with properly configured mocks."""
        # Create a sample audio array
        audio_array = np.zeros(16000)  # 1 second of silence
        
        # Mock the load_audio function
        mock_whisperx.load_audio.return_value = audio_array
        
        # Mock the transcription model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello world", "id": 0}
            ]
        }
        mock_whisperx.load_model.return_value = mock_model
        
        # Mock the alignment model and result
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_whisperx.load_align_model.return_value = (mock_align_model, mock_metadata)
        
        # Mock the align function to return a valid structure
        mock_whisperx.align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world",
                    "id": 0,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.5, "end": 1.0}
                    ]
                }
            ]
        }
        
        # Mock the diarization pipeline instance
        mock_diarize_instance = MagicMock()
        mock_diarize_instance.return_value = {
            "segments": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}
            ]
        }
        mock_diarize_pipeline_class.return_value = mock_diarize_instance
        
        # Mock the assign_word_speakers function
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world",
                    "id": 0,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                        {"word": "world", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}
                    ]
                }
            ]
        }
        
        # Mock open function for file writing
        with patch('builtins.open', unittest.mock.mock_open()) as mock_open: # type: ignore
            # Process the audio - use the audio array directly instead of a file path
            success, output_file = self.processor.process_audio(
                "dummy_path.wav", audio_array=audio_array
            )
            
            # Check results
            self.assertTrue(success)
            self.assertIsNotNone(output_file)
            
            # Verify function calls
            mock_whisperx.load_model.assert_called_once()
            mock_model.transcribe.assert_called_once()
            mock_whisperx.load_align_model.assert_called_once()
            mock_whisperx.align.assert_called_once()
            mock_whisperx.assign_word_speakers.assert_called_once()
            
            # Verify diarization pipeline was called correctly
            mock_diarize_pipeline_class.assert_called_once()
            mock_diarize_instance.assert_called_once()

# Run the test if this file is executed directly
if __name__ == "__main__":
    unittest.main()
