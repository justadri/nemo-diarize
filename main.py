# filename: main.py

import os
import logging
from pathlib import Path
from tqdm import tqdm
import requests
from dotenv import load_dotenv

import assemblyai as aai

# Import our modules
from audio_preprocessor import AudioPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = './output'
AUDIO_OUT_DIR = os.path.join (OUT_DIR, 'audio')
TRANSCRIPT_OUT_DIR = os.path.join(OUT_DIR, 'transcripts')

load_dotenv()
aai.settings.api_key = os.getenv("aai_api_key")

def check_dependencies():
    # """Check required dependencies."""
    # try:
    #     # Check ffmpeg-python
    #     try:
    #         import ffmpeg
    #         logger.info("ffmpeg-python is already installed.")
    #     except ImportError:
    #         logger.error("ffmpeg-python is not installed.")
    #         return False

    #     return True
    # except Exception as e:
    #     logger.error(f"Error checking dependencies: {str(e)}")
    #     return False
    return True

def convert_timestamp(milliseconds):
    seconds, milliseconds = divmod(milliseconds, 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}')

def main():
    # Check dependencies
    if not check_dependencies():
        logger.error("Required dependencies not found. Exiting.")
        return

    # create output folders if they don't exist
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_OUT_DIR, exist_ok=True)

    # Initialize processors
    audio_preprocessor = AudioPreprocessor()

    logger.info("Here we go!")
    test_mode = (os.getenv('ND_TEST_MODE', '0') != '0')
    if test_mode:
        logger.warning("In test mode, using default parameters")

    # Get the directory to process from the user
    input_audio_dir = "recordings" if test_mode else input("Enter the directory containing audio files to process: ")

    # Check if the directory exists
    if not os.path.isdir(input_audio_dir):
        logger.error(f"Directory not found: {input_audio_dir}")
        return

    # Get language for transcription
    language = "en" if test_mode else (input(
        "Enter language code for transcription (e.g., 'en' for English, default is English): ").strip() or "en")

    # Show available preprocessing profiles
    profiles = audio_preprocessor.get_available_profiles()
    print("\nAvailable Audio Preprocessing Profiles:")
    for name, description in profiles.items():
        print(f"- {name}: {description}")

    # Get preprocessing profile
    profile_name = "telephone" if test_mode else (input(
        "\nSelect preprocessing profile (default: standard): ").strip().lower() or "standard")
    if profile_name not in profiles:
        logger.warning(f"Profile '{profile_name}' not found. Using 'standard' profile.")
        profile_name = "standard"

    # Get all audio files in the directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.amr', '.wma', '.aac', '.aiff', '.alac', '.opus']
    input_audio_files = []

    for ext in audio_extensions:
        input_audio_files.extend(list(Path(input_audio_dir).glob(f"*{ext}")))

    if not input_audio_files:
        logger.warning(f"No audio files found in {input_audio_dir}")
        return

    logger.info(f"Found {len(input_audio_files)} audio files to process")

    input_audio_files.sort()

    # Confifure transcriber
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        format_text=True,
        punctuate=True,
        speech_model=aai.SpeechModel.slam_1,
        language_code="en_us",
    )

    transcriber = aai.Transcriber(config=config)

    # Process each audio file
    for input_audio_file in tqdm(input_audio_files, desc="Processing audio files", colour='blue'):
        input_audio_path = str(input_audio_file)
        logger.info(f"Processing {input_audio_path}")

        base_name = (os.path.splitext(os.path.basename(input_audio_path))[0]).replace(' ', '_')
        combined_results_path = os.path.join(TRANSCRIPT_OUT_DIR, f"{base_name}.txt")

        if os.path.exists(combined_results_path) and os.path.getsize(combined_results_path) > 0:
            logger.info(f"already processed {input_audio_file}, moving on")
            continue

        processed_audio_output_path = os.path.join(AUDIO_OUT_DIR, base_name + '.wav')

        if not os.path.exists(processed_audio_output_path):
            # Step 1: Preprocess audio with selected profile
            processed_audio_path, sample_rate = audio_preprocessor.preprocess_audio(
                input_file=input_audio_path,
                profile_name=profile_name,
                output_path=processed_audio_output_path,
            )
            if processed_audio_path is None or sample_rate is None:
                logger.error(f"Failed to preprocess audio: {input_audio_path}")
                continue
        else:
            processed_audio_path = processed_audio_output_path

        # step 2: upload the file
        response = requests.post(url='https://api.assemblyai.com/v2/upload',
                                 headers={'Authorization': aai.settings.api_key,
                                          "Content-Type": "application/octet-stream"},
                                 data=open(processed_audio_path, 'rb')
                                 )

        response.raise_for_status()

        upload_location = response.json()['upload_url']

        # step 3: transcribe
        logger.info(f"starting transcription of {input_audio_file}")
        transcript = transcriber.transcribe(upload_location)
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(transcript.error)

        if not isinstance(transcript.utterances, list):
            logger.error("no utterances found")
            continue
        with open(combined_results_path, 'w') as file:
            file.write(f"# transcript of {input_audio_file}\n")
            file.write(f"# {'-' * 50}\n")
            for utterance in transcript.utterances:
                line = f"[{convert_timestamp(utterance.start)}] [speaker {utterance.speaker}]: {utterance.text}\n"
                file.write(line)
        logger.info(f'completed transcription of {input_audio_file}')

    logger.info("All files processed. Results are in the ./diarization_results, ./transcription_results, and ./combined_results directories.")

if __name__ == "__main__":
    main()
