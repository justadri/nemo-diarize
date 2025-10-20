# filename: utils.py

import os
import logging
import tempfile
import soundfile as sf

logger = logging.getLogger(__name__)

# this can probably be deleted
def save_audio_to_temp(audio_array, sample_rate):
    """
    Save audio array to a temporary file.
    
    Args:
        audio_array (numpy.ndarray): Audio data as numpy array
        sample_rate (int): Sample rate of the audio
        
    Returns:
        str: Path to the temporary file
    """
    try:
        # Use NamedTemporaryFile without delete=False to avoid deletion issues
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_name = temp_file.name
        temp_file.close()  # Close the file but keep it on disk
        sf.write(temp_name, audio_array, sample_rate)
        return temp_name
    except Exception as e:
        logger.error(f"Error saving audio to temp file: {str(e)}")
        return None

# shouid use or delte this too
def format_timestamp(seconds):
    """
    Format seconds into a readable timestamp.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted timestamp as MM:SS.MS
    """
    minutes = int(seconds // 60)
    seconds_remainder = int(seconds % 60)
    milliseconds = int((seconds % 1) * 100)
    return f"{minutes:02d}:{seconds_remainder:02d}.{milliseconds:02d}"