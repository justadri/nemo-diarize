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
            
            # Create the config for the diarizer
            cfg_dict = {
                "diarizer": {
                    "manifest_filepath": "manifest.json",
                    "out_dir": "./diarization_results",
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
                            "threshold": 0.5
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
            # Create manifest file
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
            
            manifest_path = "manifest.json"
            with open(manifest_path, "w") as f:
                f.write(json.dumps(manifest))
            
            logger.info(f"Created manifest file for {audio_file_path}")
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
            os.makedirs("./diarization_results", exist_ok=True)
            
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
                return False, None
            
            # Update the manifest path in the config directly
            # FIX: This is the line that was causing the error
            self.cfg.diarizer.manifest_filepath = manifest_path
            
            # Run diarization
            logger.info(f"Running NeMo diarization on {audio_file_path}")
            self.diarizer.diarize()
            
            # The results are saved to the output directory specified in the config
            base_name = os.path.basename(audio_file_path).split('.')[0]
            result_file = os.path.join("./diarization_results", f"{base_name}_diar_rttm.txt")
            
            # Check if the result file exists
            if os.path.exists(result_file):
                logger.info(f"NeMo diarization completed successfully. Results saved to {result_file}")
                return True, result_file
            else:
                logger.warning(f"NeMo diarization completed but no result file found at {result_file}")
                return False, None
            
        except Exception as e:
            logger.error(f"Error processing file with NeMo: {str(e)}")
            return False, None
