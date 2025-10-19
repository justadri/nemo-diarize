# filename: test_nemo_processor.py

import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
import sys
from unittest.mock import patch, MagicMock, mock_open

# Create a mock for the entire nemo module and its submodules
sys.modules['nemo'] = MagicMock()
sys.modules['nemo.collections'] = MagicMock()
sys.modules['nemo.collections.asr'] = MagicMock()
sys.modules['nemo.collections.asr.models'] = MagicMock()

# Import the NemoProcessor
from nemo_processor import NemoProcessor

class TestNemoProcessor(unittest.TestCase):
    """Test cases for the NemoProcessor class."""
    
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
        
        # Create test directories
        os.makedirs("./diarization_results", exist_ok=True)
        
        # Create a test audio array
        self.audio_array = np.zeros(16000)  # 1 second of silence at 16kHz
        self.sample_rate = 16000
    
    def test_initialize_diarizer(self):
        """Test diarizer initialization."""
        # Create a mock for the ClusteringDiarizer
        mock_diarizer_class = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_diarizer_class.return_value = mock_diarizer_instance
        
        # Replace the actual import with our mock
        with patch.dict('sys.modules', {
            'nemo.collections.asr.models': MagicMock(ClusteringDiarizer=mock_diarizer_class)
        }):
            # Create a new instance of NemoProcessor which will use our mock
            processor = NemoProcessor()
            
            # Verify that our mock was used
            self.assertIsNotNone(processor.diarizer)
    
    @patch('nemo.collections.asr.models.ClusteringDiarizer')
    def test_create_manifest(self, mock_diarizer):
        """Test manifest creation."""
        mock_instance = MagicMock()
        mock_diarizer.return_value = mock_instance
        
        processor = NemoProcessor()
            
        # Mock open to avoid actually writing to a file
        with patch('builtins.open', mock_open()) as mock_file:
            manifest_path = processor.create_manifest(self.test_audio_path, 3.0)
            
            self.assertIsNotNone(manifest_path)
            mock_file.assert_called_once_with("manifest.json", "w")
            
            # Check that the write method was called with JSON content
            handle = mock_file()
            write_call_args = handle.write.call_args[0][0]
            self.assertIn(os.path.abspath(self.test_audio_path), write_call_args)
            self.assertIn("3.0", write_call_args)  # Duration as string in JSON
    
    @patch('nemo.collections.asr.models.ClusteringDiarizer')
    def test_process_audio(self, mock_diarizer):
        """Test audio processing with mocked diarizer."""
        mock_instance = MagicMock()
        mock_diarizer.return_value = mock_instance
        
        processor = NemoProcessor()
        
        # Mock the diarize method
        processor.diarizer.diarize = MagicMock() # type: ignore
        
        # Mock the create_manifest method
        processor.create_manifest = MagicMock(return_value="manifest.json")
        
        # Create a fake RTTM output file
        base_name = os.path.basename(self.test_audio_path).split('.')[0]
        rttm_output = os.path.join("./diarization_results", f"{base_name}_diar_rttm.txt")
        
        # Mock os.path.exists to return True for our output file
        with patch('os.path.exists', return_value=True), \
             patch('soundfile.info', return_value=MagicMock(duration=3.0)):
            
            # Process the audio
            success, output_file = processor.process_audio(
                self.test_audio_path, self.audio_array, self.sample_rate
            )
            
            self.assertTrue(success)
            self.assertEqual(output_file, rttm_output)
            processor.create_manifest.assert_called_once()
            processor.diarizer.diarize.assert_called_once() # type: ignore

if __name__ == "__main__":
    unittest.main()
