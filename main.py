
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: main.py

import os
import logging
import torch
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# Import our modules
from audio_preprocessor import AudioPreprocessor
from whisperx_processor import WhisperXProcessor
from nemo_processor import NemoProcessor
from result_merger import ResultMerger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check required dependencies."""
    try:
        # Check WhisperX
        try:
            import whisperx
            logger.info("WhisperX is already installed.")
        except ImportError:
            logger.error("WhisperX is not installed.")
            return False
        
        # Check ffmpeg-python
        try:
            import ffmpeg
            logger.info("ffmpeg-python is already installed.")
        except ImportError:
            logger.error("ffmpeg-python is not installed.")
            return False

        return True
    except Exception as e:
        logger.error(f"Error checking dependencies: {str(e)}")
        return False

def main():
    """Main function to process all audio files in a directory."""
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Using CPU. This may be slow.")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Required dependencies not found. Exiting.")
        return
    
    # Initialize processors
    audio_preprocessor = AudioPreprocessor()
    whisperx_processor = WhisperXProcessor()
    nemo_processor = NemoProcessor()
    result_merger = ResultMerger()
    
    # Get the directory to process from the user
    audio_dir = "recordings" #input("Enter the directory containing audio files to process: ")
    
    # Check if the directory exists
    if not os.path.isdir(audio_dir):
        logger.error(f"Directory not found: {audio_dir}")
        return
    
    # Get language for transcription
    language = "en" # input("Enter language code for transcription (e.g., 'en' for English, default is English): ").strip() or "en"
    
    # Show available preprocessing profiles
    profiles = audio_preprocessor.get_available_profiles()
    print("\nAvailable Audio Preprocessing Profiles:")
    for name, description in profiles.items():
        print(f"- {name}: {description}")
    
    # Get preprocessing profile
    profile_name = "telephone" # input("\nSelect preprocessing profile (default: standard): ").strip().lower() or "standard"
    if profile_name not in profiles:
        logger.warning(f"Profile '{profile_name}' not found. Using 'standard' profile.")
        profile_name = "standard"
    
    # Get all audio files in the directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.amr', '.wma', '.aac', '.aiff', '.alac', '.opus']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(list(Path(audio_dir).glob(f"*{ext}")))
    
    if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio_path = str(audio_file)
        logger.info(f"Processing {audio_path}")
        
        # Step 1: Preprocess audio with selected profile
        audio_array, sample_rate = audio_preprocessor.preprocess_audio(audio_path, profile_name)
        
        if audio_array is None or sample_rate is None:
            logger.error(f"Failed to preprocess audio: {audio_path}")
            raise Exception("Preprocessing failed")

        # Step 3: Process with WhisperX for transcription
        whisperx_success, whisperx_file, raw_text = whisperx_processor.process_audio(audio_path, audio_array, language)

        # Step 2: Process with NeMo for diarization
        nemo_success, nemo_file = nemo_processor.process_audio(audio_file_path=audio_path, audio_array=audio_array, sample_rate=sample_rate, raw_text=raw_text)
        
        # Step 4: Merge results if both were successful
        if nemo_success and whisperx_success:
            merge_success, combined_file = result_merger.merge_results(audio_path, nemo_file, whisperx_file)
            if merge_success:
                logger.info(f"Successfully processed {audio_path}")
            else:
                logger.warning(f"Failed to merge results for {audio_path}")
        else:
            logger.warning(f"Failed to process {audio_path} with {'NeMo' if not nemo_success else ''}  {'WhisperX' if not whisperx_success else ''}")
    
    logger.info("All files processed. Results are in the ./diarization_results, ./transcription_results, and ./combined_results directories.")

if __name__ == "__main__":
    main()