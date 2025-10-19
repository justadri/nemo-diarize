
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: audio_preprocessor.py

import os
import logging
import numpy as np
import ffmpeg

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Class for preprocessing audio using FFmpeg with specialized profiles."""
    
    # Preprocessing profiles for different recording types
    PROFILES = {
        "standard": {
            "description": "General purpose profile with balanced settings",
            "filters": {
                "noise_reduction": {"enabled": True, "strength": 0.3, "smoothing": 0.95},
                "highpass": {"enabled": True, "frequency": 80},
                "loudnorm": {"enabled": True, "target_i": -23, "lra": 7, "tp": -2}
            }
        },
        "telephone": {
            "description": "Optimized for telephone calls with narrow frequency range and compression",
            "filters": {
                "bandpass": {"enabled": True, "frequency": 300, "width": 3400},
                "noise_reduction": {"enabled": True, "strength": 0.4, "smoothing": 0.9},
                "compand": {"enabled": True, "attack": 0.02, "decay": 0.2, "soft_knee": 6, "gain": 5},
                "highpass": {"enabled": False},
                "loudnorm": {"enabled": True, "target_i": -18, "lra": 5, "tp": -1.5}
            }
        },
        "noisy": {
            "description": "Enhanced noise reduction for recordings with significant background noise",
            "filters": {
                "noise_reduction": {"enabled": True, "strength": 0.6, "smoothing": 0.85},
                "highpass": {"enabled": True, "frequency": 100},
                "afftdn": {"enabled": True, "noise_reduction": 12, "noise_floor": -50},
                "loudnorm": {"enabled": True, "target_i": -23, "lra": 5, "tp": -2},
                "equalizer": {"enabled": True, "frequency": 1000, "width": 1, "gain": 3}
            }
        },
        "conference": {
            "description": "Optimized for conference room recordings with echo and multiple speakers",
            "filters": {
                "noise_reduction": {"enabled": True, "strength": 0.4, "smoothing": 0.95},
                "highpass": {"enabled": True, "frequency": 100},
                "dereverberation": {"enabled": True},
                "loudnorm": {"enabled": True, "target_i": -23, "lra": 7, "tp": -2}
            }
        },
        "outdoor": {
            "description": "Handles wind noise and variable background sounds in outdoor recordings",
            "filters": {
                "highpass": {"enabled": True, "frequency": 120},
                "noise_reduction": {"enabled": True, "strength": 0.5, "smoothing": 0.9},
                "dynaudnorm": {"enabled": True, "frame_len": 500, "gaussian": 15},
                "loudnorm": {"enabled": True, "target_i": -23, "lra": 10, "tp": -2}
            }
        },
        "extreme_telephone": {
            "description": "Aggressive processing for very poor quality telephone audio",
            "filters": {
                "bandpass": {"enabled": True, "frequency": 250, "width": 3500},  # Wider band for more natural sound
                "noise_reduction": {"enabled": True, "strength": 0.5, "smoothing": 0.85},  # Stronger noise reduction
                "compand": {"enabled": True, "attack": 0.01, "decay": 0.15, "soft_knee": 8, "gain": 7},  # More aggressive compression
                "equalizer": {"enabled": True, "frequency": 2500, "width": 1.5, "gain": 5},  # Enhance clarity
                "loudnorm": {"enabled": True, "target_i": -16, "lra": 4, "tp": -1.5}  # Even louder normalization
            }
        },
        "extreme_noise": {
            "description": "Maximum noise reduction for extremely noisy environments",
            "filters": {
                "noise_reduction": {"enabled": True, "strength": 0.7, "smoothing": 0.8},  # Maximum noise reduction
                "afftdn": {"enabled": True, "noise_reduction": 15, "noise_floor": -60},  # More aggressive FFT denoising
                "highpass": {"enabled": True, "frequency": 120},  # Higher cutoff to remove more rumble
                "equalizer": {"enabled": True, "frequency": 1500, "width": 2, "gain": 6},  # Stronger speech enhancement
                "loudnorm": {"enabled": True, "target_i": -20, "lra": 4, "tp": -2}  # Tighter dynamic range
            }
        }
    }

    def __init__(self):
        """Initialize the audio preprocessor."""
        pass
    
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
        try:
            logger.info(f"Preprocessing audio: {input_file} with profile: {profile_name}")
            
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
            stream = ffmpeg.input(input_file)  
            
            # Apply audio filters in sequence
            filter_chain = []
            
            # Bandpass filter (especially for telephone)
            if "bandpass" in profile["filters"] and profile["filters"]["bandpass"]["enabled"]:
                bp_settings = profile["filters"]["bandpass"]
                freq = bp_settings.get("frequency", 300)
                width = bp_settings.get("width", 3400)
                filter_chain.append(f"bandpass=f={freq}:width_type=h:w={width}")
                logger.info(f"Applied bandpass filter ({freq}Hz-{freq+width}Hz)")
            
            # High-pass filter to remove low rumble
            if "highpass" in profile["filters"] and profile["filters"]["highpass"]["enabled"]:
                hp_settings = profile["filters"]["highpass"]
                freq = hp_settings.get("frequency", 80)
                filter_chain.append(f"highpass=f={freq}")
                logger.info(f"Applied high-pass filter ({freq}Hz)")
            
            # Noise reduction
            if "noise_reduction" in profile["filters"] and profile["filters"]["noise_reduction"]["enabled"]:
                nr_settings = profile["filters"]["noise_reduction"]
                strength = nr_settings.get("strength", 0.3)
                smoothing = nr_settings.get("smoothing", 0.95)
                filter_chain.append(f"anlmdn=s={strength}:p={smoothing}")
                logger.info(f"Applied noise reduction filter (strength={strength}, smoothing={smoothing})")
            
            # FFT-based noise reduction (for very noisy recordings)
            if "afftdn" in profile["filters"] and profile["filters"]["afftdn"]["enabled"]:
                fft_settings = profile["filters"]["afftdn"]
                nr = fft_settings.get("noise_reduction", 12)
                nf = fft_settings.get("noise_floor", -50)
                filter_chain.append(f"afftdn=nr={nr}:nf={nf}")
                logger.info(f"Applied FFT-based noise reduction (reduction={nr}dB, floor={nf}dB)")
            
            # Dynamic audio normalization
            if "dynaudnorm" in profile["filters"] and profile["filters"]["dynaudnorm"]["enabled"]:
                dyn_settings = profile["filters"]["dynaudnorm"]
                frame_len = dyn_settings.get("frame_len", 500)
                gaussian = dyn_settings.get("gaussian", 15)
                filter_chain.append(f"dynaudnorm=f={frame_len}:g={gaussian}")
                logger.info(f"Applied dynamic audio normalization (frame_len={frame_len}, gaussian={gaussian})")
            
            # Compressor/expander for telephone
            if "compand" in profile["filters"] and profile["filters"]["compand"]["enabled"]:
                comp_settings = profile["filters"]["compand"]
                attack = comp_settings.get("attack", 0.02)
                decay = comp_settings.get("decay", 0.2)
                soft_knee = comp_settings.get("soft_knee", 6)
                gain = comp_settings.get("gain", 5)
                # Compand filter format: attack,decay soft-knee points gain
                filter_chain.append(f"compand={attack},{decay} {soft_knee}:-70/-60,-20/-10,0/-5 {gain}")
                logger.info(f"Applied compressor/expander (attack={attack}s, decay={decay}s, gain={gain}dB)")
            
            # Dereverberation for conference rooms
            if "dereverberation" in profile["filters"] and profile["filters"]["dereverberation"]["enabled"]:
                # Use areverse,afftdn,areverse trick for dereverberation
                filter_chain.append("areverse,afftdn=nr=10:nf=-40,areverse")
                logger.info("Applied dereverberation filter")
            
            # Equalizer for frequency adjustments
            if "equalizer" in profile["filters"] and profile["filters"]["equalizer"]["enabled"]:
                eq_settings = profile["filters"]["equalizer"]
                freq = eq_settings.get("frequency", 1000)
                width = eq_settings.get("width", 1)
                gain = eq_settings.get("gain", 3)
                filter_chain.append(f"equalizer=f={freq}:width_q={width}:g={gain}")
                logger.info(f"Applied equalizer (frequency={freq}Hz, width={width}, gain={gain}dB)")
            
            # Audio normalization
            if "loudnorm" in profile["filters"] and profile["filters"]["loudnorm"]["enabled"]:
                ln_settings = profile["filters"]["loudnorm"]
                target_i = ln_settings.get("target_i", -23)
                lra = ln_settings.get("lra", 7)
                tp = ln_settings.get("tp", -2)
                filter_chain.append(f"loudnorm=I={target_i}:LRA={lra}:TP={tp}")
                logger.info(f"Applied loudness normalization (target={target_i}LUFS, range={lra}, peak={tp}dB)")
            
            # Apply all filters if any were specified
            if filter_chain:
                filter_string = ','.join(filter_chain)
                stream = stream.filter(filter_string)
            
            # Set output format to 16kHz mono WAV (optimal for speech recognition)
            stream = stream.output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=16000)
            
            # Run the ffmpeg process and capture output
            out, _ = stream.run(capture_stdout=True, quiet=True)
            
            # Convert the output bytes to a numpy array
            audio_array = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
            
            logger.info(f"Audio preprocessing completed successfully: {input_file}")
            return audio_array, 16000  # Return audio array and sample rate
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
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