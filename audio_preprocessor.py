import os
import sys
import logging
import numpy as np
import subprocess
import re
import ffmpeg

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Class for preprocessing audio using FFmpeg with specialized profiles."""
    SAMPLE_RATE = 16000
    
    # Preprocessing profiles for different recording types
    # noise reduction strength 0.00001 to 10000, smoothing 1 to 1000
    PROFILES = {
        "standard": {
            "description": "General purpose profile with balanced settings",
            "filters": {
                "highpass": {"frequency": 60},
                "dynaudnorm": {"frame_length": 500, "gauss_size": 31, "peak": 0.95, "max_gain": 5.0}
                # "noise_reduction": {"strength": 0.1, "smoothing": 10},
                # "highpass": {"frequency": 80},
                # "loudnorm": {"target_i": -23, "lra": 7, "tp": -2}
            }
        },
        "telephone": {
            "description": "Optimized for telephone calls with narrow frequency range and compression",
            "filters": {
                "volume_adapt": {"target_mean": -10},
                "bandpass": {"frequency": 2050, "width": 1950},
                "equalizer": {
                              "bands": [
                                  {"frequency": 200, "width": 1, "gain": 2},
                                  {"frequency": 1000, "width": 1, "gain": 3},
                                  {"frequency": 3000, "width": 1, "gain": 2}
                                ]
                              },
                "dynaudnorm": {"frame_length": 100, "gauss_size": 15, "peak": 0.95, "max_gain": 10.0},
                "acompressor": {"threshold": 0.125, "ratio": 2, "attack": 75, "release": 400, "makeup": 4},
                "alimiter": {"level_in": 0.9, "level_out": 0.9, "limit": 1.0, "attack": 5, "release": 50}
            }
        },
        "noisy": {
            "description": "Optimized for nolisy environments",
            "filters": {
                "volume_adapt": {"target_mean": -10},
                "bandpass": {"frequency": 4040, "width": 3060},
                "afftdn": {"noise_floor": -20},
                "equalizer": {
                    "bands": [
                        {"frequency": 1000, "width": 1, "gain": 2},
                        {"frequency": 2000, "width": 1, "gain": 3},
                        {"frequency": 3000, "width": 1, "gain": 2}
                    ]
                },
                "compand": {
                    "attacks": "0.01", 
                    "decays": "0.5", 
                    "points": "-90/-90|-60/-40|-40/-30|-30/-20|0/-10",
                    "gain": 5
                },
                "loudnorm": {
                    "integrated_target": -6,
                    "range_target": 11,
                    "max_true_peak": -1
                }
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
    
    def preprocess_audio(self, input_file, profile_name="standard", custom_filters=None, output_path=None):
        """
        Preprocess audio using ffmpeg with a specific profile or custom filters.
        Returns the audio as a numpy array ready for processing.
        
        Args:
            input_file (str): Path to the input audio file
            profile_name (str): Name of the preprocessing profile to use
            custom_filters (dict, optional): Custom filter settings to override profile
            output_path (str): the path (with filename) to store the processed file. if none, returns a numpy array
        Returns:
            tuple: (union(audio_array|audio_path), sample_rate) or (None, None) if processing fails
        """
        if not self.ffmpeg_available:
            logger.error("FFmpeg is not available. Cannot preprocess audio.")
            return None, None

        """Implementation using ffmpeg-python."""
        try:
            logger.info(f"Preprocessing audio with ffmpeg-python: {input_file}")
            
            # Verify the input file exists
            abs_path_in = os.path.abspath(input_file)
            if not os.path.exists(abs_path_in):
                logger.error(f"Input file does not exist: {abs_path_in}")
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
                stream = ffmpeg.input(abs_path_in)
            except Exception as e:
                logger.error(f"Failed to create FFmpeg input: {str(e)}")
                return None, None
            
            # Apply audio filters in sequence
            filter_chain = []
            
            if "volume_adapt" in profile["filters"]:
                logger.info("Applying volume adaptation filter")
                logger.info("Analyzing volume levels...")
                detected_mean_volume = None
                try:
                    out, err = (
                        ffmpeg.input(abs_path_in)
                        .filter('volumedetect')
                        .output('/dev/null', format='null')
                        .run(quiet=True, capture_stderr=True, capture_stdout=True)
                    )
                    # Extract max_volume from stderr
                    match = re.search(r'mean_volume: ([-+]?\d*\.?\d+) dB', err.decode('utf-8'))
                    detected_mean_volume = float(match.group(1)) if match else None
                except ffmpeg.Error as e:
                    logger.error(f"Error analyzing volume levels: {str(e)}, {e.stderr.decode('utf-8')}")
                
                if detected_mean_volume is not None:
                    target_level = profile["filters"]["volume_adapt"].get("target_mean", -10)
                    volume_adjustment = target_level - detected_mean_volume
                    logger.info(f"Detected mean volume: {detected_mean_volume} dB, adjusting by {volume_adjustment} dB"
                                +" to reach target {target_level} dB")
                    filter_chain.append({"name": "volume", "args": { "volume": f"{volume_adjustment}dB" }})
                    
            # Bandpass filter (especially for telephone)
            if "bandpass" in profile["filters"]:
                bp_settings = profile["filters"]["bandpass"]
                freq = bp_settings.get("frequency", 790)
                width = bp_settings.get("width", 730)
                filter_chain.append({"name": "bandpass", "args": { "f": freq, "width_type": "h", "w": width }})
                logger.info(f"Applied bandpass filter ({freq-width}Hz-{freq+width}Hz)")
            
            # High-pass filter to remove low rumble
            if "highpass" in profile["filters"]:
                hp_settings = profile["filters"]["highpass"]
                freq = hp_settings.get("frequency", 80)
                filter_chain.append({"name": "highpass", "args": { "f": freq }})
                logger.info(f"Applied high-pass filter ({freq}Hz)")
            
            # Noise reduction
            if "noise_reduction" in profile["filters"]:
                nr_settings = profile["filters"]["noise_reduction"]
                strength = nr_settings.get("strength", 100)
                smoothing = nr_settings.get("smoothing", 1)
                # Use a more compatible filter for testing
                filter_chain.append({"name": "anlmdn", "args": { "strength": int(strength), "smooth": int(smoothing) }})
                # filter_chain.append(f"afftdn=nr=10:nf=-20")
                logger.info(f"Applied noise reduction filter")
            
            # FFT-based noise reduction (for very noisy recordings)
            if "afftdn" in profile["filters"]:
                fft_settings = profile["filters"]["afftdn"]
                nr = fft_settings.get("noise_reduction", 12)
                nf = fft_settings.get("noise_floor", -50)
                filter_chain.append({"name": "afftdn", "args": { "nr": nr, "nf": nf }})
                # filter_chain.append(f"afftdn=nr={nr}:nf={nf}")
                logger.info(f"Applied FFT-based noise reduction")
            
            # Equalizer filter
            if "equalizer" in profile["filters"]:
                eq_settings = profile["filters"]["equalizer"]
                for band in eq_settings.get("bands", []):
                    freq = band.get("frequency", 1000)
                    width = band.get("width", 100)
                    gain = band.get("gain", 0)
                    filter_chain.append({"name": "equalizer", "args": { "f": freq, "w": width, "g": gain }})
                    logger.info(f"Applied equalizer band ({freq}Hz, {width}Hz, {gain}dB)")
            
            # Dynamic audio normalization
            if "dynaudnorm" in profile["filters"]:
                dan_settings = profile["filters"]["dynaudnorm"]
                frame_length = dan_settings.get("frame_length", 500)
                gauss_size = dan_settings.get("gauss_size", 31)
                peak = dan_settings.get("peak", 0.95)
                max_gain = dan_settings.get("max_gain", 5.0)
                filter_chain.append({"name": "dynaudnorm", "args": {
                    "framelen": frame_length, 
                    "gausssize": gauss_size, 
                    "peak": peak, 
                    "maxgain": max_gain 
                }})
                logger.info(f"Applied dynamic audio normalization")

            # Audio normalization
            if "loudnorm" in profile["filters"]:
                ln_settings = profile["filters"]["loudnorm"]
                target_i = ln_settings.get("integrated_target", -23)
                lra = ln_settings.get("range_target", 7)
                tp = ln_settings.get("max_true_peak", -2)
                filter_chain.append({"name": "loudnorm", "args": { "I": target_i, "LRA": lra, "TP": tp }})  
                # filter_chain.append(f"loudnorm=I={target_i}:LRA={lra}:TP={tp}")
                logger.info(f"Applied loudness normalization")
                
            # Dynamic range compression
            if "compand" in profile["filters"]:
                comp_settings = profile["filters"]["compand"]
                attacks = comp_settings.get("attacks", 0.02)
                decays = comp_settings.get("decays", 0.2)
                points = comp_settings.get("points", "-70/-70|-60/-20|1/0")
                soft_knee = comp_settings.get("soft_knee", 0.01)
                gain = comp_settings.get("gain", 0)
                filter_chain.append({"name": "compand", "args": { 
                    "attacks": attacks,
                    "decays": decays,
                    "points": points,
                    "soft-knee": soft_knee,
                    "gain": gain 
                }})
                # filter_chain.append(f"compand=attack={attack}:decay={decay}:soft_knee={soft_knee}:gain={gain}")
                logger.info(f"Applied dynamic range compression")
            
            # compressor
            if "acompressor" in profile["filters"]:
                ac_settings = profile["filters"]["acompressor"]
                threshold = ac_settings.get("threshold", 0.1)
                ratio = ac_settings.get("ratio", 2)
                attack = ac_settings.get("attack", 50)
                release = ac_settings.get("release", 250)
                makeup = ac_settings.get("makeup", 3)
                filter_chain.append({"name": "acompressor", "args": { "threshold": threshold, "ratio": ratio, "attack": attack, "release": release, "makeup": makeup }})
                logger.info(f"Applied audio compressor")
                
            # limiter
            if "alimiter" in profile["filters"]:
                al_settings = profile["filters"]["alimiter"]
                level_in = al_settings.get("level_in", 0.9)
                level_out = al_settings.get("level_out", 0.9)
                limit = al_settings.get("limit", 1.0)
                attack = al_settings.get("attack", 5)
                release = al_settings.get("release", 50)
                filter_chain.append({"name": "alimiter", "args": { "level_in": level_in, "level_out": level_out, "limit": limit, "attack": attack, "release": release }})
                logger.info(f"Applied audio limiter")
            
            # Apply all filters if any were specified
            if filter_chain:
                try:
                    for filter in filter_chain:
                        stream = stream.filter(filter_name=filter["name"], **filter["args"])
                except Exception as e:
                    logger.error(f"Failed to apply filters: {str(e)}")
                    # Fall back to no filters
                    stream = ffmpeg.input(abs_path_in)
            
            # Set output format to 16kHz mono WAV
            output_channel = 'pipe:'
             
            abs_path_out = ''
            if output_path:
                abs_path_out = os.path.abspath(output_path)    
                if os.path.exists(os.path.dirname(abs_path_out)):
                    output_channel = abs_path_out
                else:
                    logger.error(f"output directory does not exist: {abs_path_out}")
            logger.info(f"outputting audio to {output_channel}")
            
            try:
                stream = stream.output(output_channel, format='wav', acodec='pcm_s16le', ac=1, ar=self.SAMPLE_RATE, 
                                       hide_banner=None, loglevel="error")
            except Exception as e:
                logger.error(f"Failed to set output format: {str(e)}")
                return None, None
            
            # Run the ffmpeg process and capture output
            try:
                # # Print the FFmpeg command for debugging
                # cmd = ffmpeg.compile(stream)
                # logger.info(f"FFmpeg command: {' '.join(cmd)}")
                
                out, err = stream.run(capture_stdout=True, capture_stderr=True, quiet=False)
                
                if err:
                    logger.warning(f"FFmpeg stderr output: {err.decode('utf-8')}")
                
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg error: {str(e)}")
                if e.stderr:
                    logger.error(f"FFmpeg stderr: {e.stderr.decode('utf-8')}")
                # Try fallback to simple conversion
                return self.preprocess_audio_simple(input_file, output_path)
            except Exception as e:
                logger.error(f"Error running FFmpeg: {str(e)}")
                # Try fallback to simple conversion
                return self.preprocess_audio_simple(input_file, output_path)
            
            # Convert the output bytes to a numpy array
            if output_channel == 'pipe:':
                try:
                    audio_array = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
                    
                    if len(audio_array) == 0:
                        logger.error("FFmpeg produced empty output")
                        return None, None
                    
                    logger.info(f"Audio preprocessing completed successfully: {input_file}")
                    return audio_array, self.SAMPLE_RATE  # Return audio array and sample rate
                except Exception as e:
                    logger.error(f"Error converting FFmpeg output to numpy array: {str(e)}")
                    return None, None
                
            else:
                return abs_path_out, self.SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            # Try fallback to simple conversion
            return self.preprocess_audio_simple(input_file)
    
    def preprocess_audio_simple(self, input_file, output_path):
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
            abs_path_in = os.path.abspath(input_file)
            if not os.path.exists(abs_path_in):
                logger.error(f"Input file does not exist: {abs_path_in}")
                return None, None
            
            out_dest = '-'
            
            if output_path:
                abs_path_out = os.path.abspath(output_path)
                if not os.path.exists(os.path.dirname(abs_path_out)):
                    logger.error(f"output directory does not exist: {abs_path_out}")
                    return None, None
                else:
                    out_dest = abs_path_out
                    
            
            # Use subprocess for simplicity and reliability
            cmd = [
                'ffmpeg',
                '-i', abs_path_in,
                '-f', 'wav',
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', str(self.SAMPLE_RATE),
                out_dest
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
                return None, None
            
            if out_dest == '-':    
                # Convert the output bytes to a numpy array
                audio_array = np.frombuffer(process.stdout, np.int16).astype(np.float32) / 32768.0
            
                if len(audio_array) == 0:
                    logger.error("FFmpeg produced empty output")
                    return None, None
                
                logger.info(f"Simple preprocessing completed successfully: {input_file}")
                return audio_array, self.SAMPLE_RATE
            else:
                logger.info(f"Simple preprocessing completed successfully: {input_file}")
                return abs_path_out, self.SAMPLE_RATE
            
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
