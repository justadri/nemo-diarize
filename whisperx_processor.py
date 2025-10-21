# filename: whisperx_processor.py

import os
import json
import re
from typing import Any
import torch
import logging
import gc
import numpy as np
import traceback
from whisperx.asr import FasterWhisperPipeline


logger = logging.getLogger(__name__)
OUT_DIR = "./output/whisperx"
LANGUAGE = 'en'

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
        os.environ['PYANNOTE_CACHE'] = os.path.abspath('./models')
        self.asr_model:FasterWhisperPipeline = self.initialize_asr()
        self.align_model, self.align_metadata = self.initialize_align()
        self.diarize_model = self.initialize_diarize()
        
    def initialize_asr(self):
        logger.info("Initializing WhisperX ASR...")
        # Create output directory if it doesn't exist
        os.makedirs(OUT_DIR, exist_ok=True)
        model = whisperx.load_model("large-v3-turbo", self.device, 
                                    compute_type=self.compute_type,
                                    vad_method="silero",
                                    vad_options={
                                        'model': 'snakers4/silero-vad',
                                        'min_silence_duration_ms': 500,
                                        'min_speech_duration_ms': 250,
                                        'threshold': 0.3 # Try values between 0.3-0.7
                                    },
                                    download_root='./models', 
                                    language=LANGUAGE,
                                    asr_options={
                                        'multilingual': False,
                                        'hallucination_silence_threshold': 10,
                                        'no_speech_threshold': 0.05,
                                        'beam_size': 5,
                                        'word_timestamps': True,
                                        'temperatures': [0.0, 0.2, 0.4],
                                        # 'vad_filter': True,
                                        'compression_ratio_threshold': 2.8, # Adjust if hallucinating
                                        'condition_on_previous_text': True,
                                        'initial_prompt': "this is a conversation about medical concerns", # Add domain context'
                                    },
                                    local_files_only=False
                                    )
        logger.info("WhisperX ASR loaded successfully")
        return model
    
    def initialize_align(self):
        logger.info("Initializing WhisperX alignment model...")
        model, metadata = whisperx.load_align_model(language_code=LANGUAGE, 
                                            device=self.device,
                                            model_name='facebook/wav2vec2-large-960h-lv60-self',
                                            # model_name='SrihariGKS/wav2vec-asr-fine-tuned-english-3',
                                            model_dir='./models'
                                            )
        logger.info("WhisperX alignment model loaded successfully")
        return model, metadata
    
    def initialize_diarize(self):
        logger.info("Initializing WhisperX diarization model...")
        model = DiarizationPipeline(use_auth_token=os.getenv('HUGGINGFACE_TOKEN'),
                                    device=self.device)
        logger.info("WhisperX diarization model loaded successfully")
        return model
  
    def normalize_text(self, text):
        """
        Normalizes text by fixing common ASR issues.
        """
        if not text:
            return text
            
        # Convert to string if not already
        text = str(text)
        
        # Fix common ASR errors
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        text = text.strip()
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])\s+([.,!?;:])', r'\1\2', text)  # Fix double punctuation
        
        # Fix capitalization
        sentences = re.split(r'([.!?])\s+', text)
        result = ""
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].capitalize()
                result += sentence
                
            if i + 1 < len(sentences):
                result += sentences[i+1] + " "
        
        # Handle edge case where split didn't work (no sentence terminators)
        if not result:
            result = text.capitalize()
            
        # Fix common transcription errors (customize based on your domain)
        # replacements = {
        #     "gonna": "going to",
        #     "wanna": "want to",
        #     "kinda": "kind of",
        #     # Add domain-specific replacements here
        # }
        
        # for original, replacement in replacements.items():
        #     result = re.sub(r'\b' + original + r'\b', replacement, result, flags=re.IGNORECASE)
        
        return result

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
            # 1. Transcribe with Whisper (batched)
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
            
            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]

            logger.info(f"Running WhisperX transcription on {audio_file_path}...")
            # Run transcription
            asr_results = self.asr_model.transcribe(
                audio,
                batch_size=16,
                language=language,
                print_progress=True
            )
            with open(os.path.join(OUT_DIR, f"{base_name}_asr_raw.json"), "w") as f:
                json.dump(asr_results, f, indent=2)
                
            # 2. Align the transcription
            logger.info("Aligning whisper output...")
            aligned_results = whisperx.align(
                asr_results["segments"], 
                self.align_model, 
                self.align_metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            with open(os.path.join(OUT_DIR, f"{base_name}_aligned_raw.json"), "w") as f:
                json.dump(aligned_results, f, indent=2)

            # 3. Diarize with speaker diarization model
            logger.info("Starting diarization...")
            diarize_segments = self.diarize_model(
                audio, 
                min_speakers=2, 
                max_speakers=8
            )
            with open(os.path.join(OUT_DIR, f"{base_name}_diarize_raw.json"), "w") as f:
                json.dump(diarize_segments.to_dict(), f, indent=2)
            
            # 4. Assign speaker labels
            logger.info("Assigning speaker labels...")
            labeled_results =  whisperx.assign_word_speakers(diarize_segments, aligned_results)
            
            apply_text_normalization=True
            merge_short_segments=True
            filter_low_confidence=False
            confidence_threshold=0.3
            min_segment_duration=0.5
            """
            Applies post-processing to improve the quality of the final output.
            
            Args:
                aligned_result: List of segments with speaker information
                apply_text_normalization: Whether to normalize text
                merge_short_segments: Whether to merge very short segments
                filter_low_confidence: Whether to filter low confidence segments
                confidence_threshold: Minimum confidence score to keep a segment
                min_segment_duration: Minimum duration for a segment in seconds
            
            Returns:
                Processed list of segments
            """
            processed_segments = []
            
            # Filter by confidence if needed
            if filter_low_confidence:
                filtered_segments = []
                for segment in labeled_results['segments']:
                    if not 'confidence' in segment or segment['confidence'] >= confidence_threshold:
                        filtered_segments.append(segment)
                    # Optionally mark low confidence segments instead of removing
                    else:
                        segment['text'] = f"[Low confidence: {segment['text']}]"
                        filtered_segments.append(segment)
            else:
                filtered_segments = labeled_results['segments']
            
            # Merge short segments if needed
            if merge_short_segments:
                merged_segments = []
                current_segment = None
                
                for segment in filtered_segments:
                    # Start a new current segment if none exists
                    if current_segment is None:
                        current_segment = segment
                        continue
                        
                    # Check if segments can be merged (same speaker and short)
                    can_merge = (
                        'speaker' in current_segment and
                        'speaker' in segment and
                        current_segment['speaker'] == segment['speaker'] and
                        (segment['end'] - segment['start']) < min_segment_duration
                    )
                    
                    # Check for time proximity
                    time_proximity = (segment['start'] - current_segment['end']) < 0.5
                    
                    if can_merge and time_proximity:
                        # Merge the segments
                        current_segment['end'] = segment['end']
                        current_segment['text'] = f"{current_segment['text']} {segment['text']}"
                        
                        # Merge word timestamps if available
                        if 'words' in current_segment and 'words' in segment:
                            current_segment['words'].extend(segment['words'])
                    else:
                        # Add the current segment and start a new one
                        merged_segments.append(current_segment)
                        current_segment = segment
                        
                # Add the last segment if it exists
                if current_segment is not None:
                    merged_segments.append(current_segment)
                    
                result_segments = merged_segments
            else:
                result_segments = filtered_segments
            
            # Apply text normalization if needed
            if apply_text_normalization:
                for segment in result_segments:
                    segment['text'] = self.normalize_text(segment['text'])
            
            # Final cleanup and formatting
            for segment in result_segments:
                # Round timestamps for cleaner output
                segment['start'] = round(segment['start'], 3)
                segment['end'] = round(segment['end'], 3)
                
                # Ensure all segments have required attributes
                if not 'speaker' in segment:
                    segment['speaker'] = "UNKNOWN"
                    
                processed_segments.append(segment)

            logger.info("WhisperX completed successfully, saving results...")
            # Save the results
            output_file = os.path.join(OUT_DIR, f"{base_name}_transcript.json")
            
            output = {"segments": processed_segments}
            
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
                       
            logger.info(f"WhisperX processing completed. Results saved to {output_file}") # and {readable_output}")
            return True, output_file
            
        except Exception as e:
            logger.error(f"Error processing file with WhisperX: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
