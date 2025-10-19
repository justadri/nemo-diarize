
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: whisperx_processor.py

import os
import json
import torch
import logging
import gc

logger = logging.getLogger(__name__)

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
            import whisperx
            
            # Create output directory if it doesn't exist
            os.makedirs("./transcription_results", exist_ok=True)
            
            logger.info(f"Running WhisperX transcription on {audio_file_path}")
            
            # 1. Transcribe with Whisper (batched)
            model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)
            
            # Use preprocessed audio if provided, otherwise load from file
            if audio_array is not None:
                audio = audio_array
            else:
                audio = whisperx.load_audio(audio_file_path)
            
            # Run transcription
            result = model.transcribe(audio, batch_size=16, language=language)
            
            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=language, device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            
            # 3. Assign speaker labels
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=self.device)
            diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=8)
            
            # 4. Assign speaker to words
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Save the results
            base_name = os.path.basename(audio_file_path).split('.')[0]
            output_file = os.path.join("./transcription_results", f"{base_name}_transcript.json")
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Also save a readable format
            readable_output = os.path.join("./transcription_results", f"{base_name}_transcript.txt")
            
            with open(readable_output, 'w') as f:
                f.write(f"Transcription results for {audio_file_path}\n")
                f.write("=" * 50 + "\n\n")
                
                current_speaker = None
                transcript_text = ""
                
                for segment in result["segments"]:
                    for word in segment.get("words", []):
                        speaker = word.get("speaker", "UNKNOWN")
                        
                        if current_speaker is None:
                            current_speaker = speaker
                            transcript_text = word["word"] + " "
                        elif speaker != current_speaker:
                            f.write(f"[Speaker {current_speaker}]: {transcript_text.strip()}\n\n")
                            current_speaker = speaker
                            transcript_text = word["word"] + " "
                        else:
                            transcript_text += word["word"] + " "
                
                if transcript_text:
                    f.write(f"[Speaker {current_speaker}]: {transcript_text.strip()}\n")
            
            # Clean up GPU memory
            del model, model_a, diarize_model
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"WhisperX processing completed. Results saved to {output_file} and {readable_output}")
            return True, output_file
            
        except Exception as e:
            logger.error(f"Error processing file with WhisperX: {str(e)}")
            return False, None