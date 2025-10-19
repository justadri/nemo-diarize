# filename: test_nemo_processor.py

import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
from unittest.mock import patch, MagicMock
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
    
    @patch('nemo.collections.asr.models.ClusteringDiarizer')
    def test_initialize_diarizer(self, mock_diarizer):
        """Test diarizer initialization."""
        mock_instance = MagicMock()
        mock_diarizer.return_value = mock_instance
        
        processor = NemoProcessor()
        
        mock_diarizer.assert_called_once()
        self.assertEqual(processor.diarizer, mock_instance)
    
    @patch('nemo.collections.asr.models.ClusteringDiarizer')
    def test_create_manifest(self, mock_diarizer):
        """Test manifest creation."""
        mock_instance = MagicMock()
        mock_diarizer.return_value = mock_instance
        
        processor = NemoProcessor()
        manifest_path = processor.create_manifest(self.test_audio_path, 3.0)
        
        self.assertIsNotNone(manifest_path)
        self.assertTrue(os.path.exists(manifest_path))
        
        # Verify manifest content
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.assertEqual(manifest["audio_filepath"], os.path.abspath(self.test_audio_path))
        self.assertEqual(manifest["duration"], 3.0)
    
    @patch('nemo.collections.asr.models.ClusteringDiarizer')
    def test_process_audio(self, mock_diarizer):
        """Test audio processing with mocked diarizer."""
        mock_instance = MagicMock()
        mock_diarizer.return_value = mock_instance
        
        processor = NemoProcessor()
        
        # Mock the diarize method
        processor.diarizer.diarize = MagicMock()
        
        # Mock the create_manifest method
        processor.create_manifest = MagicMock(return_value="manifest.json")
        
        # Create a fake RTTM output file
        base_name = os.path.basename(self.test_audio_path).split('.')[0]
        rttm_output = os.path.join("./diarization_results", f"{base_name}_diar_rttm.txt")
        with open(rttm_output, 'w') as f:
            f.write("SPEAKER test 1 0.0 1.0 <NA> <NA> SPEAKER_00 <NA> <NA>\n")
        
        # Process the audio
        success, output_file = processor.process_audio(
            self.test_audio_path, self.audio_array, self.sample_rate
        )
        
        self.assertTrue(success)
        self.assertEqual(output_file, rttm_output)
        processor.create_manifest.assert_called_once()
        processor.diarizer.diarize.assert_called_once()
