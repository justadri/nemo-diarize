# filename: whisperx_processor.py

import os
import json
from typing import Any
import torch
import logging
import gc
import numpy as np


logger = logging.getLogger(__name__)
OUT_DIR = "./whisperx_results"

# Import DiarizationPipeline conditionally to handle testing scenarios
DiarizationPipeline:Any = None
from whisperx.diarize import DiarizationPipeline as _DiarizationPipeline # type: ignore
DiarizationPipeline = _DiarizationPipeline

# Import whisperx here to allow for mocking in tests
whisperx:Any = None
import whisperx as _whisperx  # type: ignore
whisperx = _whisperx


class WhisperXProcessor:
    """Class for processing audio with WhisperX."""
    
    def __init__(self):
        """Initialize the WhisperX processor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    def process_audio(self, audio_file_path, audio_array=None, language="en"):
        """
        Process audio with WhisperX for transcription and diarization.
        
        Args:
            audio_file_path (str): Path to the audio file (used for output naming)
            audio_array (numpy.ndarray, optional): Preprocessed audio array
            language (str): Language code for transcription
            
        Returns:
            bool: True if processing was successful, False otherwise
            str: Path to the output JSON file if successful, None otherwise
        """
        try:            
            # Create output directory if it doesn't exist
            os.makedirs(OUT_DIR, exist_ok=True)
            
            logger.info(f"Running WhisperX transcription on {audio_file_path}")
            
            # 1. Transcribe with Whisper (batched)
            model = whisperx.load_model("distil-large-v3", self.device, compute_type=self.compute_type)
            
            # Use preprocessed audio if provided, otherwise load from file
            if audio_array is not None:
                audio = audio_array
                logger.info("Using provided audio array")
            else:
                # Check if file exists before trying to load it
                if os.path.exists(audio_file_path):
                    audio = whisperx.load_audio(audio_file_path)
                    logger.info(f"Loaded audio from file: {audio_file_path}")
                else:
                    raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Run transcription
            result = model.transcribe(audio, batch_size=16, language=language)
            
            # print(result)  # Debug: print the raw transcription result
            
            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=language, device=self.device)
            
            # Add error handling for align model
            if hasattr(model_a, "__class__") and hasattr(model_a.__class__, "__name__"):
                logger.info(f"Loaded align model of type: {model_a.__class__.__name__}")
            else:
                logger.warning(f"Align model type: {type(model_a)}")
                
            # Check if we're in a test environment with mocks
            is_mock = hasattr(model_a, "_extract_mock_name") or str(type(model_a)).find("MagicMock") >= 0 or model_a.__class__.__name__ == "AlignModel"
            
            aligned_segments = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            
            result = aligned_segments

            # 3. Assign speaker labels
            diarize_model = DiarizationPipeline(use_auth_token=None, device=self.device)
            
            diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=8)
            
            # 4. Assign speaker to words
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Save the results
            base_name = os.path.basename(audio_file_path).split(".")[0]
            output_file = os.path.join(OUT_DIR, f"{base_name}_transcript.json")
            
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            # Also save a readable format
            readable_output = os.path.join(OUT_DIR, f"{base_name}_transcript.txt")
            
            with open(readable_output, "w") as f:
                f.write(f"Transcription results for {audio_file_path}\n")
                f.write("=" * 50 + "\n\n")
                
                current_speaker = None
                
                for segment in result["segments"]:
                    if "speaker" in segment and "text" in segment:
                        f.write(f"[Speaker {segment['speaker']}]: {segment['text']}\n\n")
            
            # Clean up GPU memory
            del model, model_a
            if diarize_model is not None and not is_mock:
                del diarize_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"WhisperX processing completed. Results saved to {output_file} and {readable_output}")
            return True, output_file
            
        except Exception as e:
            logger.error(f"Error processing file with WhisperX: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, None
