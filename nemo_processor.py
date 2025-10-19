import os
import json
import torch
import logging
import soundfile as sf
import tempfile
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

class NemoProcessor:
    """Class for processing audio with NeMo diarization."""
    
    def __init__(self):
        """Initialize the NeMo processor."""
        # Store config separately from diarizer
        self.cfg = None
        self.diarizer = self.initialize_diarizer()
    
    def initialize_diarizer(self):
        """Initialize the NeMo ClusteringDiarizer model."""
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            
            logger.info("Initializing NeMo ClusteringDiarizer model...")
            
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Create temp directory if it doesn't exist
            temp_dir = "./temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create output directory if it doesn't exist
            out_dir = "./diarization_results"
            os.makedirs(out_dir, exist_ok=True)
            
            # Create a temporary manifest file to ensure it exists
            manifest_path = os.path.join(temp_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                f.write("[]")  # Write an empty JSON array as a placeholder
            
            # Create the config for the diarizer
            cfg_dict = {
                "num_workers": 0,
                "max_num_of_spks": 8,
                "scale_n": 5,
                "soft_label_thres": 0.5,
                "emb_batch_size": 0,
                "sample_rate": 16000,
                "verbose": True,
                "diarizer": {
                    "manifest_filepath": manifest_path,
                    "out_dir": out_dir,
                    "speaker_embeddings": {
                        "model_path": "titanet_large",
                        "parameters": {
                            "window_length_in_sec": 1.5,
                            "shift_length_in_sec": 0.75,
                            "multiscale_weights": None,
                            "save_embeddings": False
                        }
                    },
                    "clustering": {
                        "parameters": {
                            "oracle_num_speakers": False,
                            "max_num_speakers": 8,
                            "enhanced_count_thres": 80,
                            "max_rp_threshold": 0.25,
                            "sparse_search_volume": 30
                        }
                    },
                    "vad": {
                        "model_path": "vad_multilingual_marblenet",
                        "parameters": {
                            "window_length_in_sec": 0.15,
                            "shift_length_in_sec": 0.01,
                            "threshold": 0.5,
                            "smoothing": "median", # False or type of smoothing method (eg: median)
                            "overlap": 0.25,
                            "onset": 0.4, # Onset threshold for detecting the beginning and end of a speech
                            "offset": 0.7, # Offset threshold for detecting the end of a speech
                            "pad_onset": 0.05, # Adding durations before each speech segment
                            "pad_offset": -0.1, # Adding durations after each speech segment
                            "min_duration_on": 0.2, # Threshold for short speech segment deletion
                            "min_duration_off": 0.2, # Threshold for small non_speech deletion
                            "filter_speech_first": True
                        }
                    },
                    "oracle_vad": False,
                    "collar": 0.25,
                    "ignore_overlap": False
                },
                "device": device
            }
            
            # Convert to OmegaConf and store it separately
            self.cfg = OmegaConf.create(cfg_dict)
            
            # Initialize the diarizer with our config
            diarizer = ClusteringDiarizer(cfg=self.cfg)
            
            logger.info("NeMo ClusteringDiarizer model initialized successfully.")
            return diarizer
        except Exception as e:
            logger.error(f"Error initializing diarizer: {str(e)}")
            return None
    
    def create_manifest(self, audio_file_path, duration):
        """Create a manifest file for the audio file."""
        try:
            # Create temp directory if it doesn't exist
            temp_dir = "./temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create manifest object
            manifest = {
                "audio_filepath": os.path.abspath(audio_file_path),
                "offset": 0,
                "duration": duration,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": None,
                "uem_filepath": None
            }

            manifest_path = os.path.join(temp_dir, "manifest.json")
            
            # Write as JSONL (one JSON object per line)
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(manifest) + "\n")
            
            # Verify the file was written correctly
            if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
                logger.error(f"Failed to write manifest file or file is empty")
                return None
            
            # Read back and log the exact content for debugging
            with open(manifest_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Manifest file content: {content}")
            
            logger.info(f"Created valid manifest file for {audio_file_path}")
            return manifest_path
        except Exception as e:
            logger.error(f"Error creating manifest for {audio_file_path}: {str(e)}")
            return None
    
    def process_audio(self, audio_file_path, audio_array=None, sample_rate=16000):
        """
        Process audio with NeMo for diarization.
        
        Args:
            audio_file_path (str): Path to the audio file
            audio_array (numpy.ndarray, optional): Preprocessed audio array
            sample_rate (int): Sample rate of the audio array
            
        Returns:
            bool: True if processing was successful, False otherwise
            str: Path to the output RTTM file if successful, None otherwise
        """
        if self.diarizer is None or self.cfg is None:
            logger.error("NeMo diarizer is not initialized")
            return False, None
            
        try:
            # Create output directory if it doesn't exist
            out_dir = "./diarization_results"
            os.makedirs(out_dir, exist_ok=True)
            
            # If preprocessed audio is provided, save it to a temporary file for NeMo
            temp_file = None
            if audio_array is not None:
                # Use NamedTemporaryFile without delete=False to avoid deletion issues
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_name = temp_file.name
                temp_file.close()  # Close the file but keep it on disk
                sf.write(temp_name, audio_array, sample_rate)
                audio_file_path_for_nemo = temp_name
                duration = len(audio_array) / sample_rate
            else:
                audio_file_path_for_nemo = audio_file_path
                audio_info = sf.info(audio_file_path)
                duration = audio_info.duration
            
            # Create manifest for this audio file
            manifest_path = self.create_manifest(audio_file_path_for_nemo, duration)
            if not manifest_path:
                logger.error("Failed to create manifest file")
                return False, None
            
            # Update the manifest path in the config directly
            self.cfg.diarizer.manifest_filepath = manifest_path
            
            # Run diarization
            logger.info(f"Running NeMo diarization on {audio_file_path}")
            
            # Verify the manifest file exists and is valid before diarizing
            if not os.path.exists(manifest_path):
                logger.error(f"Manifest file {manifest_path} does not exist")
                return False, None
                
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_content = f.read()
                    logger.info(f"Manifest content before diarization: {manifest_content}")
                    # Verify it's valid JSON
                    json.loads(manifest_content)
            except Exception as e:
                logger.error(f"Error reading manifest before diarization: {str(e)}")
                return False, None
            
            # Now run diarization
            self.diarizer.diarize()
            
            # The results are saved to the output directory specified in the config
            base_name = os.path.basename(audio_file_path).split('.')[0]
            result_file = os.path.join(out_dir, f"{base_name}_diar_rttm.txt")
            
            # Check if the result file exists
            if os.path.exists(result_file):
                logger.info(f"NeMo diarization completed successfully. Results saved to {result_file}")
                return True, result_file
            else:
                logger.warning(f"NeMo diarization completed but no result file found at {result_file}")
                return False, None
            
        except Exception as e:
            logger.error(f"Error processing file with NeMo: {str(e)}")
            raise e
            return False, None
