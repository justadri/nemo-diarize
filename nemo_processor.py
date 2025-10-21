import os
import json
import torch
import logging
import soundfile as sf
import tempfile
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
from nemo.core.config import hydra_runner

logger = logging.getLogger(__name__)
OUT_DIR = "./output/nemo"

class NemoProcessor:
    """Class for processing audio with NeMo diarization."""
    
    def __init__(self):
        """Initialize the NeMo processor."""
        # Store config separately from diarizer
        self.diarizer = self.initialize_diarizer()
    
    def initialize_diarizer(self):
        """Initialize NeMo."""        
        logger.info("Initializing NeMo...")
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create output directory if it doesn't exist
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'temp'), exist_ok=True)
        
        # Create a temporary manifest file to ensure it exists
        manifest_path = os.path.join(OUT_DIR, "manifest.json")
        with open(manifest_path, "w") as f:
            f.write("[]")  # Write an empty JSON array as a placeholder
        
        # Create the config for the diarizer
        path_to_arpa = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', '4gram_big.arpa')
        
        num_workers = 0
        batch_size = 8
        emb_batch_size = 0
        if device == 'cuda':
            num_workers = 0
            batch_size = 8
            emb_batch_size = 8
        
        cfg_dict = {
            "name": "ClusterDiarizer",
            "num_workers": num_workers, # Increase from 0 for parallel processing
            "sample_rate": 16000,
            "batch_size": batch_size, # Increase for better GPU utilization
            "device": device,
            "verbose": True,
            "max_num_of_spks": 8,
            "scale_n": 8, # Higher value increases sensitivity to speaker differences
            "soft_label_thres": 0.4, # Lower threshold for easier speaker separation
            "emb_batch_size": emb_batch_size, # Set positive for batch processing
            "diarizer": {
                "manifest_filepath": manifest_path,
                "out_dir": OUT_DIR,
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": False,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "external_vad_manifest": None,
                    "parameters": {
                        "window_length_in_sec": 0.5,
                        "shift_length_in_sec": 0.05,
                        "smoothing": "median", # False or type of smoothing method (eg: median)
                        "overlap": 0.5,
                        "onset": 0.4, # Onset threshold for detecting the beginning and end of a speech
                        "offset": 0.4, # Offset threshold for detecting the end of a speech
                        "pad_onset": 0.1, # Adding durations before each speech segment
                        "pad_offset": 0.1, # Adding durations after each speech segment
                        "min_duration_on": 0.15, # Threshold for short speech segment deletion
                        "min_duration_off": 0.15, # Threshold for small non_speech deletion
                        "filter_speech_first": True
                        # "threshold": 0.5
                    }
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.0, 0.5, 0.3], # Window length(s) in sec (floating-point number). either a number or a list. ex) 1.5 or [1.5,1.0,0.5]
                        "shift_length_in_sec": [0.75, 0.5, 0.25, 0.15], # Shift length(s) in sec (floating-point number). either a number or a list. ex) 0.75 or [0.75,0.5,0.25]
                        "multiscale_weights": [0.2, 0.2, 0.3, 0.3], # Weight for each scale. should be null (for single scale) or a list matched with window/shift scale count. ex) [0.33,0.33,0.33]
                        "save_embeddings": False # Save embeddings as pickle file for each audio input.
                    }
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False, # If True, use num of speakers value provided in manifest file.
                        "max_num_speakers": 8, # Max number of speakers for each recording. If an oracle number of speakers is passed, this value is ignored.
                        "enhanced_count_thres": 60, # If the number of segments is lower than this number, enhanced speaker counting is activated.
                        "max_rp_threshold": 0.15, # Determines the range of p-value search: 0 < p <= max_rp_threshold. 
                        "sparse_search_volume": 20, # The higher the number, the more values will be examined with more time. 
                        "maj_vote_spk_count": True,  # If True, take a majority vote on multiple p-values to estimate the number of speakers.
                        "chunk_cluster_count": 70, # Number of forced clusters (overclustering) per unit chunk in long-form audio clustering.
                        "embeddings_per_chunk": 8000 # Number of embeddings in each chunk for long-form audio clustering. Adjust based on GPU memory capacity. (default: 10000, approximately 40 mins of audio) 
                    }
                }
            }
        }
        
        # Convert to OmegaConf and store it separately
        self.cfg = OmegaConf.create(cfg_dict)
        
        # # Initialize the diarizer with our config
        diarizer = ClusteringDiarizer(cfg=self.cfg).to(self.cfg.device)
        
        logger.info("NeMo ClusteringDiarizer model initialized successfully.")
        
        return diarizer
    
    def create_manifest(self, audio_file_path, duration):
        """Create a manifest file for the audio file."""
        try:
            # Create manifest object
            manifest = {
                "audio_filepath": os.path.abspath(audio_file_path),
                "offset": 0,
                "duration": duration,
                "label": "infer",
                "text": "-"
            }

            manifest_path = os.path.join(OUT_DIR, "manifest.json")
            
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

        if self.cfg is None:
            logger.error("NeMo is not initialized")
            return False, None
            
        try:
            # Debug audio information
            if audio_array is not None:
                logger.debug(f"Audio array shape: {audio_array.shape}, sample rate: {sample_rate}")
                logger.debug(f"Audio duration: {len(audio_array) / sample_rate} seconds")
            else:
                logger.warning(f"Using file path: {audio_file_path}")
                if os.path.exists(audio_file_path):
                    audio_info = sf.info(audio_file_path)
                    logger.debug(f"Audio file info: {audio_info}")
                else:
                    logger.error(f"Audio file does not exist: {audio_file_path}")
                    return False, None
            
            # Process audio as before with added debugging
            temp_file = None
            if audio_array is not None:
                temp_name = os.path.join(OUT_DIR, 'temp', os.path.splitext(os.path.basename(audio_file_path))[0] + '.wav')
                sf.write(temp_name, audio_array, sample_rate)
                audio_file_path_for_nemo = temp_name
                duration = len(audio_array) / sample_rate
                logger.debug(f"Saved audio array to temp file: {temp_name}")
            else:
                audio_file_path_for_nemo = audio_file_path
                audio_info = sf.info(audio_file_path)
                duration = audio_info.duration
            
            # Create manifest with debugging
            manifest_path = self.create_manifest(audio_file_path_for_nemo, duration)
            if not manifest_path:
                logger.error("Failed to create manifest file")
                return False, None
            
            # Debug manifest content
            with open(manifest_path, "r") as f:
                manifest_content = f.read()
                logger.debug(f"Manifest content: {manifest_content}")
                logger.debug(f"Manifest file size: {os.path.getsize(manifest_path)} bytes")
            
            # Update config and debug it
            self.cfg.diarizer.manifest_filepath = manifest_path
            logger.debug(f"Diarizer config: {OmegaConf.to_yaml(self.cfg)}")
            
            # # Debug NeMo internals before diarizing
            # logger.debug("Checking NeMo dataloader setup...")
            # try:
            #     # Try to manually create the dataloader to debug
            #     from nemo.collections.asr.data.audio_to_label import AudioToSpeechLabelDataset
            #     dataset = AudioToSpeechLabelDataset(
            #         manifest_filepath=manifest_path,
            #         featurizer=self.diarizer.preprocessor,
            #         labels=[]
            #     )
            #     logger.debug(f"Dataset created successfully with {len(dataset)} items")
                
            #     # Try to get one item
            #     if len(dataset) > 0:
            #         item = dataset[0]
            #         logger.debug(f"First dataset item: {item}")
            # except Exception as e:
            #     logger.debug(f"Error testing dataloader: {str(e)}")
            
            # Diarization inference for speaker labels
            logger.info("Running NeMo diarization...")

            self.diarizer.diarize()
            logger.info("Diarization completed successfully")
            
            # Check results
            base_name = os.path.splitext(os.path.basename(audio_file_path_for_nemo))[0]
            result_file = os.path.join(OUT_DIR, "pred_rttms", f"{base_name}.rttm")
            
            if os.path.exists(result_file):
                logger.info(f"NeMo diarization completed successfully. Results saved to {result_file}")
                return True, result_file
            else:
                logger.error(f"NeMo diarization completed but no result file found at {result_file}")
                return False, None
                
        except Exception as e:
            logger.error(f"Error processing file with NeMo: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            raise e
            return False, None
