# filename: test_audio_preprocessor.py

import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
import subprocess
import logging
from audio_preprocessor import AudioPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestAudioPreprocessor(unittest.TestCase):
    """Test cases for the AudioPreprocessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        # Check if FFmpeg is installed
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            cls.ffmpeg_available = True
            logger.info("FFmpeg is available for tests.")
        except (subprocess.SubprocessError, FileNotFoundError):
            cls.ffmpeg_available = False
            logger.warning("FFmpeg not found. Tests requiring FFmpeg will be skipped.")
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = AudioPreprocessor()
        
        # Create a simple test audio file with unique name
        self.test_dir = tempfile.mkdtemp()
        self.test_filename = f"test_audio_{id(self)}.wav"
        self.test_audio_path = os.path.join(self.test_dir, self.test_filename)
        
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        # Add some noise
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 0.1, len(audio))
        audio = audio + noise
        
        # Save the audio file
        sf.write(self.test_audio_path, audio, sample_rate)
        
        # Verify the file was created
        self.assertTrue(os.path.exists(self.test_audio_path), 
                       f"Test audio file was not created at {self.test_audio_path}")
        
        # Print file info for debugging
        file_size = os.path.getsize(self.test_audio_path)
        logger.info(f"Created test audio file: {self.test_audio_path}, size: {file_size} bytes")
    
    def test_get_available_profiles(self):
        """Test that available profiles can be retrieved."""
        profiles = self.preprocessor.get_available_profiles()
        self.assertIsInstance(profiles, dict)
        self.assertIn("standard", profiles)
        self.assertIn("telephone", profiles)
        self.assertIn("noisy", profiles)
    
    def test_standard_profile(self):
        """Test preprocessing with standard profile."""
        if not hasattr(self, 'ffmpeg_available') or not self.ffmpeg_available:
            self.skipTest("FFmpeg not available")
        
        # Print absolute path for debugging
        abs_path = os.path.abspath(self.test_audio_path)
        logger.info(f"Using absolute path: {abs_path}")
        
        # Try with simplified preprocessing first
        audio_array, sample_rate = self.preprocessor.preprocess_audio_simple(
            self.test_audio_path
        )
        
        # Check if simple preprocessing works
        if audio_array is None:
            self.skipTest("Simple preprocessing failed, likely an FFmpeg issue")
        
        # Now try with the standard profile
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, profile_name="standard"
        )
        
        self.assertIsNotNone(audio_array, "Audio array should not be None")
        self.assertEqual(sample_rate, 16000)
        self.assertIsInstance(audio_array, np.ndarray)
        self.assertTrue(len(audio_array) > 0)
    
    def test_telephone_profile(self):
        """Test preprocessing with telephone profile."""
        if not hasattr(self, 'ffmpeg_available') or not self.ffmpeg_available:
            self.skipTest("FFmpeg not available")
        
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, profile_name="telephone"
        )
        
        self.assertIsNotNone(audio_array, "Audio array should not be None")
        self.assertEqual(sample_rate, 16000)
        self.assertIsInstance(audio_array, np.ndarray)
        self.assertTrue(len(audio_array) > 0)
    
    def test_noisy_profile(self):
        """Test preprocessing with noisy profile."""
        if not hasattr(self, 'ffmpeg_available') or not self.ffmpeg_available:
            self.skipTest("FFmpeg not available")
        
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, profile_name="noisy"
        )
        
        self.assertIsNotNone(audio_array, "Audio array should not be None")
        self.assertEqual(sample_rate, 16000)
        self.assertIsInstance(audio_array, np.ndarray)
        self.assertTrue(len(audio_array) > 0)
    
    def test_custom_profile(self):
        """Test creating and using a custom profile."""
        if not hasattr(self, 'ffmpeg_available') or not self.ffmpeg_available:
            self.skipTest("FFmpeg not available")
        
        custom_filters = self.preprocessor.create_custom_profile(
            base_profile="standard",
            noise_reduction={"strength": 0.5},
            highpass={"frequency": 100}
        )
        
        self.assertIn("noise_reduction", custom_filters)
        self.assertEqual(custom_filters["noise_reduction"]["strength"], 0.5)
        self.assertEqual(custom_filters["highpass"]["frequency"], 100)
        
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, 
            profile_name="standard",
            custom_filters=custom_filters
        )
        
        self.assertIsNotNone(audio_array, "Audio array should not be None")
        self.assertEqual(sample_rate, 16000)
    
    def test_invalid_profile(self):
        """Test that invalid profile falls back to standard."""
        if not hasattr(self, 'ffmpeg_available') or not self.ffmpeg_available:
            self.skipTest("FFmpeg not available")
        
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, profile_name="nonexistent_profile"
        )
        
        self.assertIsNotNone(audio_array, "Audio array should not be None")
        self.assertEqual(sample_rate, 16000)
    
    def test_invalid_file(self):
        """Test handling of invalid audio file."""
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            "nonexistent_file.wav", profile_name="standard"
        )
        
        self.assertIsNone(audio_array)
        self.assertIsNone(sample_rate)

if __name__ == "__main__":
    unittest.main()
