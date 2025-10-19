# filename: test_whisperx_processor.py

import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
from unittest.mock import patch, MagicMock
from whisperx_processor import WhisperXProcessor

class TestWhisperXProcessor(unittest.TestCase):
    """Test cases for the WhisperXProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = WhisperXProcessor()
        
        # Create a simple test audio file with unique name
        self.test_dir = tempfile.mkdtemp()
        self.test_filename = f"test_audio_{id(self)}.wav"
        self.test_audio_path = os.path.join(self.test_dir, self.test_filename)
        
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        sf.write(self.test_audio_path, audio, sample_rate)
        
        # Create test directories
        os.makedirs("./transcription_results", exist_ok=True)
    
    @patch('whisperx_processor.WhisperXProcessor.process_audio')
    def test_process_audio_with_file(self, mock_process):
        """Test processing with audio file path."""
        mock_process.return_value = (True, "output_file.json")
        
        success, output_file = self.processor.process_audio(self.test_audio_path)
        
        mock_process.assert_called_once()
        self.assertTrue(success)
        self.assertEqual(output_file, "output_file.json")
    
    @patch('whisperx.load_model')
    @patch('whisperx.load_align_model')
    @patch('whisperx.DiarizationPipeline')
    @patch('whisperx.assign_word_speakers')
    def test_mocked_whisperx_pipeline(self, mock_assign, mock_diarize, mock_align, mock_load):
        """Test the WhisperX pipeline with mocks."""
        # Mock the WhisperX model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello world"}
            ]
        }
        mock_load.return_value = mock_model
        
        # Mock the alignment model
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_align.return_value = (mock_align_model, mock_metadata)
        
        # Mock the diarization pipeline
        mock_diarize_model = MagicMock()
        mock_diarize_model.return_value = {"segments": []}
        mock_diarize.return_value = mock_diarize_model
        
        # Mock the word speaker assignment
        mock_assign.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                        {"word": "world", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}
                    ]
                }
            ]
        }
        
        # Create a test audio array
        audio_array = np.zeros(16000)  # 1 second of silence at 16kHz
        
        # Process the audio
        with patch('builtins.open', unittest.mock.mock_open()):
            success, _ = self.processor.process_audio(
                self.test_audio_path, audio_array=audio_array
            )
        
        self.assertTrue(success)
        mock_load.assert_called_once()
        mock_model.transcribe.assert_called_once()
        mock_align.assert_called_once()
        mock_diarize.assert_called_once()
        mock_assign.assert_called_once()
