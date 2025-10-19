# filename: test_audio_preprocessor.py

import unittest
import os
import numpy as np
import tempfile
import soundfile as sf
from audio_preprocessor import AudioPreprocessor

class TestAudioPreprocessor(unittest.TestCase):
    """Test cases for the AudioPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = AudioPreprocessor()
        
        # Create a simple test audio file with unique name
        self.test_dir = tempfile.mkdtemp()
        self.test_filename = f"test_audio_{id(self)}.wav"
        self.test_audio_path = os.path.join(self.test_dir, self.test_filename)
        
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate a 1kHz sine wave
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        # Add some noise
        noise = np.random.normal(0, 0.1, len(audio))
        audio = audio + noise
        sf.write(self.test_audio_path, audio, sample_rate)
    
    def test_get_available_profiles(self):
        """Test that available profiles can be retrieved."""
        profiles = self.preprocessor.get_available_profiles()
        self.assertIsInstance(profiles, dict)
        self.assertIn("standard", profiles)
        self.assertIn("telephone", profiles)
        self.assertIn("noisy", profiles)
    
    def test_standard_profile(self):
        """Test preprocessing with standard profile."""
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, profile_name="standard"
        )
        self.assertIsNotNone(audio_array)
        self.assertEqual(sample_rate, 16000)
        self.assertIsInstance(audio_array, np.ndarray)
        self.assertTrue(len(audio_array) > 0)
    
    def test_telephone_profile(self):
        """Test preprocessing with telephone profile."""
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, profile_name="telephone"
        )
        self.assertIsNotNone(audio_array)
        self.assertEqual(sample_rate, 16000)
        self.assertIsInstance(audio_array, np.ndarray)
        self.assertTrue(len(audio_array) > 0)
    
    def test_noisy_profile(self):
        """Test preprocessing with noisy profile."""
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, profile_name="noisy"
        )
        self.assertIsNotNone(audio_array)
        self.assertEqual(sample_rate, 16000)
        self.assertIsInstance(audio_array, np.ndarray)
        self.assertTrue(len(audio_array) > 0)
    
    def test_custom_profile(self):
        """Test creating and using a custom profile."""
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
        
        self.assertIsNotNone(audio_array)
        self.assertEqual(sample_rate, 16000)
    
    def test_invalid_profile(self):
        """Test that invalid profile falls back to standard."""
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            self.test_audio_path, profile_name="nonexistent_profile"
        )
        self.assertIsNotNone(audio_array)
        self.assertEqual(sample_rate, 16000)
    
    def test_invalid_file(self):
        """Test handling of invalid audio file."""
        audio_array, sample_rate = self.preprocessor.preprocess_audio(
            "nonexistent_file.wav", profile_name="standard"
        )
        self.assertIsNone(audio_array)
        self.assertIsNone(sample_rate)
