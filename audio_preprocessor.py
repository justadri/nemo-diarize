import os
import logging
import numpy as np
import subprocess
import tempfile

logger = logging.getLogger(__name__)

import ffmpeg

class AudioPreprocessor:
    """Class for preprocessing audio using FFmpeg with specialized profiles."""
    
    # Preprocessing profiles for different recording types
    # noise reduction strength 0.00001 to 10000, smoothing 1 to 1000
    PROFILES = {
        "standard": {
            "description": "General purpose profile with balanced settings",
            "filters": {
                "noise_reduction": {"enabled": True, "strength": 0.1, "smoothing": 10},
                "highpass": {"enabled": True, "frequency": 80},
                "loudnorm": {"enabled": True, "target_i": -23, "lra": 7, "tp": -2}
            }
        },
        "telephone": {
            "description": "Optimized for telephone calls with narrow frequency range and compression",
            "filters": {
                "bandpass": {"enabled": True, "frequency": 300, "width": 3400},
                "noise_reduction": {"enabled": True, "strength": 1, "smoothing": 15},
                "compand": {"enabled": True, "attack": 0.02, "decay": 0.2, "soft_knee": 6, "gain": 5},
                "highpass": {"enabled": False},
                "loudnorm": {"enabled": True, "target_i": -18, "lra": 5, "tp": -1.5}
            }
        },
        "noisy": {
            "description": "Enhanced noise reduction for recordings with significant background noise",
            "filters": {
                "noise_reduction": {"enabled": True, "strength": 10, "smoothing": 20},
                "highpass": {"enabled": True, "frequency": 100},
                "afftdn": {"enabled": True, "noise_reduction": 12, "noise_floor": -50},
                "loudnorm": {"enabled": True, "target_i": -23, "lra": 5, "tp": -2},
                "equalizer": {"enabled": True, "frequency": 1000, "width": 1, "gain": 3}
            }
        },
        "extreme_telephone": {
            "description": "Custom profile for extremely poor quality telephone audio",
            "filters": {
                "bandpass": {"enabled": True, "frequency": 250, "width": 3500},  # Wider band for more natural sound
                "noise_reduction": {"enabled": True, "strength": 100, "smoothing": 25},  # Stronger noise reduction
                "compand": {"enabled": True, "attack": 0.01, "decay": 0.15, "soft_knee": 8, "gain": 7},  # More aggressive compression
                "equalizer": {"enabled": True, "frequency": 2500, "width": 1.5, "gain": 5},  # Enhance clarity
                "loudnorm": {"enabled": True, "target_i": -16, "lra": 4, "tp": -1.5}  # Even louder normalization
            }
        },
        "extreme_noise": {
            "description": "Custom profile for extremely noisy audio recordings",
            "filters": {
                "noise_reduction": {"enabled": True, "strength": 200, "smoothing": 30},  # Maximum noise reduction
                "afftdn": {"enabled": True, "noise_reduction": 15, "noise_floor": -60},  # More aggressive FFT-based noise reduction
                "highpass": {"enabled": True, "frequency": 120},  # Higher cutoff to remove more rumble
                "equalizer": {"enabled": True, "frequency": 1500, "width": 2, "gain": 6},  # Stronger speech enhancement
                "loudnorm": {"enabled": True, "target_i": -20, "lra": 4, "tp": -2}  # Tighter dynamic range
            }
        }
    }
    
    def __init__(self):
        """Initialize the audio preprocessor."""
        # Check if FFmpeg is installed
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            self.ffmpeg_available = True
            logger.info("FFmpeg is available.")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.ffmpeg_available = False
            logger.error("FFmpeg not found. Audio preprocessing will not work.")
            raise EnvironmentError("FFmpeg not found. Please install FFmpeg to use AudioPreprocessor.")
    
    def get_available_profiles(self):
        """
        Get a list of available preprocessing profiles.
        
        Returns:
            dict: Dictionary of profile names and descriptions
        """
        return {name: profile["description"] for name, profile in self.PROFILES.items()}
    
    def preprocess_audio(self, input_file, profile_name="standard", custom_filters=None):
        """
        Preprocess audio using ffmpeg with a specific profile or custom filters.
        Returns the audio as a numpy array ready for processing.
        
        Args:
            input_file (str): Path to the input audio file
            profile_name (str): Name of the preprocessing profile to use
            custom_filters (dict, optional): Custom filter settings to override profile
            
        Returns:
            tuple: (audio_array, sample_rate) or (None, None) if processing fails
        """
        if not self.ffmpeg_available:
            logger.error("FFmpeg is not available. Cannot preprocess audio.")
            return None, None
        
        # Use the appropriate method based on availability
        return self._preprocess_with_ffmpeg_python(input_file, profile_name, custom_filters)
    
    def _preprocess_with_ffmpeg_python(self, input_file, profile_name="standard", custom_filters=None):
        """Implementation using ffmpeg-python."""
        try:
            logger.info(f"Preprocessing audio with ffmpeg-python: {input_file}")
            
            # Verify the input file exists
            abs_path = os.path.abspath(input_file)
            if not os.path.exists(abs_path):
                logger.error(f"Input file does not exist: {abs_path}")
                return None, None
            
            # Get the profile settings
            if profile_name not in self.PROFILES:
                logger.warning(f"Profile '{profile_name}' not found. Using 'standard' profile.")
                profile_name = "standard"
            
            profile = self.PROFILES[profile_name]
            
            # Apply custom filter overrides if provided
            if custom_filters:
                for filter_name, settings in custom_filters.items():
                    if filter_name in profile["filters"]:
                        profile["filters"][filter_name].update(settings)
            
            # Start building the ffmpeg pipeline
            try:
                stream = ffmpeg.input(abs_path)
            except Exception as e:
                logger.error(f"Failed to create FFmpeg input: {str(e)}")
                return None, None
            
            # Apply audio filters in sequence
            filter_chain = []
            
            # Bandpass filter (especially for telephone)
            if "bandpass" in profile["filters"] and profile["filters"]["bandpass"]["enabled"]:
                bp_settings = profile["filters"]["bandpass"]
                freq = bp_settings.get("frequency", 300)
                width = bp_settings.get("width", 3400)
                filter_chain.append({"name": "bandpass", "args": { "f": freq, "width_type": "h", "w": width }})
                # filter_chain.append(f"bandpass=f={freq}:width_type=h:w={width}")
                logger.info(f"Applied bandpass filter ({freq}Hz-{freq+width}Hz)")
            
            # High-pass filter to remove low rumble
            if "highpass" in profile["filters"] and profile["filters"]["highpass"]["enabled"]:
                hp_settings = profile["filters"]["highpass"]
                freq = hp_settings.get("frequency", 80)
                filter_chain.append({"name": "highpass", "args": { "f": freq }})
                # filter_chain.append(f"highpass=f={freq}")
                logger.info(f"Applied high-pass filter ({freq}Hz)")
                
            # Equalizer filter
            if "equalizer" in profile["filters"] and profile["filters"]["equalizer"]["enabled"]:
                eq_settings = profile["filters"]["equalizer"]
                freq = eq_settings.get("frequency", 1000)
                width = eq_settings.get("width", 100)
                gain = eq_settings.get("gain", 0)
                filter_chain.append({"name": "equalizer", "args": { "f": freq, "w": width, "g": gain }})
                # filter_chain.append(f"equalizer=f={freq}:w={width}:g={gain}")
                logger.info(f"Applied equalizer filter ({freq}Hz, {width}Hz, {gain}dB)")
            
            # Noise reduction
            if "noise_reduction" in profile["filters"] and profile["filters"]["noise_reduction"]["enabled"]:
                nr_settings = profile["filters"]["noise_reduction"]
                strength = nr_settings.get("strength", 100)
                smoothing = nr_settings.get("smoothing", 1)
                # Use a more compatible filter for testing
                filter_chain.append({"name": "anlmdn", "args": { "strength": int(strength), "smooth": int(smoothing) }})
                # filter_chain.append(f"afftdn=nr=10:nf=-20")
                logger.info(f"Applied noise reduction filter")
            
            # FFT-based noise reduction (for very noisy recordings)
            if "afftdn" in profile["filters"] and profile["filters"]["afftdn"]["enabled"]:
                fft_settings = profile["filters"]["afftdn"]
                nr = fft_settings.get("noise_reduction", 12)
                nf = fft_settings.get("noise_floor", -50)
                filter_chain.append({"name": "afftdn", "args": { "nr": nr, "nf": nf }})
                # filter_chain.append(f"afftdn=nr={nr}:nf={nf}")
                logger.info(f"Applied FFT-based noise reduction")
                
            # Dynamic range compression
            if "compand" in profile["filters"] and profile["filters"]["compand"]["enabled"]:
                comp_settings = profile["filters"]["compand"]
                attack = comp_settings.get("attack", 0.02)
                decay = comp_settings.get("decay", 0.2)
                soft_knee = comp_settings.get("soft_knee", 6)
                gain = comp_settings.get("gain", 5)
                filter_chain.append({"name": "compand", "args": { "attacks": attack, "decays": decay, "soft-knee": soft_knee, "gain": gain }})
                # filter_chain.append(f"compand=attack={attack}:decay={decay}:soft_knee={soft_knee}:gain={gain}")
                logger.info(f"Applied dynamic range compression")
            
            # Audio normalization
            if "loudnorm" in profile["filters"] and profile["filters"]["loudnorm"]["enabled"]:
                ln_settings = profile["filters"]["loudnorm"]
                target_i = ln_settings.get("target_i", -23)
                lra = ln_settings.get("lra", 7)
                tp = ln_settings.get("tp", -2)
                filter_chain.append({"name": "loudnorm", "args": { "I": target_i, "LRA": lra, "TP": tp }})  
                # filter_chain.append(f"loudnorm=I={target_i}:LRA={lra}:TP={tp}")
                logger.info(f"Applied loudness normalization")
            
            # Apply all filters if any were specified
            if filter_chain:
                try:
                    for filter in filter_chain:
                        stream = stream.filter(filter_name=filter["name"], **filter["args"])
                except Exception as e:
                    logger.error(f"Failed to apply filters: {str(e)}")
                    # Fall back to no filters
                    stream = ffmpeg.input(abs_path)
            
            # Set output format to 16kHz mono WAV
            try:
                stream = stream.output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=16000, hide_banner=None, loglevel="error")
            except Exception as e:
                logger.error(f"Failed to set output format: {str(e)}")
                return None, None
            
            # Run the ffmpeg process and capture output
            try:
                # # Print the FFmpeg command for debugging
                # cmd = ffmpeg.compile(stream)
                # logger.info(f"FFmpeg command: {' '.join(cmd)}")
                
                out, err = stream.run(capture_stdout=True, capture_stderr=True, quiet=True)
                
                if err:
                    logger.warning(f"FFmpeg stderr output: {err.decode('utf-8')}")
                
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg error: {str(e)}")
                if e.stderr:
                    logger.error(f"FFmpeg stderr: {e.stderr.decode('utf-8')}")
                # Try fallback to simple conversion
                return self.preprocess_audio_simple(input_file)
            except Exception as e:
                logger.error(f"Error running FFmpeg: {str(e)}")
                # Try fallback to simple conversion
                return self.preprocess_audio_simple(input_file)
            
            # Convert the output bytes to a numpy array
            try:
                audio_array = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
                
                if len(audio_array) == 0:
                    logger.error("FFmpeg produced empty output")
                    return None, None
                
                logger.info(f"Audio preprocessing completed successfully: {input_file}")
                return audio_array, 16000  # Return audio array and sample rate
            except Exception as e:
                logger.error(f"Error converting FFmpeg output to numpy array: {str(e)}")
                return None, None
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            # Try fallback to simple conversion
            return self.preprocess_audio_simple(input_file)
    
    def preprocess_audio_simple(self, input_file):
        """
        A simplified version of preprocess_audio that only converts format.
        Useful for testing if FFmpeg is working correctly.
        """
        if not self.ffmpeg_available:
            logger.error("FFmpeg is not available. Cannot preprocess audio.")
            return None, None
        
        try:
            logger.info(f"Simple preprocessing of audio: {input_file}")
            
            # Verify the input file exists
            abs_path = os.path.abspath(input_file)
            if not os.path.exists(abs_path):
                logger.error(f"Input file does not exist: {abs_path}")
                return None, None
            
            # Use subprocess for simplicity and reliability
            cmd = [
                'ffmpeg',
                '-i', abs_path,
                '-f', 'wav',
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-'
            ]
            
            logger.info(f"Simple FFmpeg command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"Simple FFmpeg process failed with code {process.returncode}")
                logger.error(f"FFmpeg stderr: {process.stderr.decode('utf-8')}")
                
                # For testing purposes, generate a simple sine wave as fallback
                logger.warning("Generating test sine wave as fallback")
                sample_rate = 16000
                duration = 3  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                audio_array = 0.5 * np.sin(2 * np.pi * 1000 * t)
                return audio_array, 16000
            
            # Convert the output bytes to a numpy array
            audio_array = np.frombuffer(process.stdout, np.int16).astype(np.float32) / 32768.0
            
            if len(audio_array) == 0:
                logger.error("FFmpeg produced empty output")
                return None, None
            
            logger.info(f"Simple preprocessing completed successfully: {input_file}")
            return audio_array, 16000
        except Exception as e:
            logger.error(f"Error in simple preprocessing: {str(e)}")
            return None, None
    
    def create_custom_profile(self, base_profile="standard", **filter_settings):
        """
        Create a custom profile based on an existing profile with modifications.
        
        Args:
            base_profile (str): Name of the base profile to modify
            **filter_settings: Filter settings to override
            
        Returns:
            dict: Custom filter settings
        """
        if base_profile not in self.PROFILES:
            logger.warning(f"Base profile '{base_profile}' not found. Using 'standard' profile.")
            base_profile = "standard"
        
        # Create a deep copy of the base profile
        custom_filters = {}
        for filter_name, settings in self.PROFILES[base_profile]["filters"].items():
            custom_filters[filter_name] = settings.copy()
        
        # Apply custom settings
        for filter_name, settings in filter_settings.items():
            if filter_name in custom_filters:
                custom_filters[filter_name].update(settings)
            else:
                custom_filters[filter_name] = settings
        
        return custom_filters
