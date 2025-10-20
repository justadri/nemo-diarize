
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: result_merger.py

import os
import json
import logging

logger = logging.getLogger(__name__)

class ResultMerger:
    """Class for merging results from WhisperX and NeMo."""
    
    def __init__(self):
        """Initialize the result merger."""
        pass
    
    def merge_results(self, audio_file_path, nemo_file, whisperx_file):
        """
        Merge NeMo diarization with WhisperX transcription for a comprehensive result.
        
        Args:
            audio_file_path (str): Path to the original audio file
            nemo_file (str): Path to the NeMo RTTM file
            whisperx_file (str): Path to the WhisperX JSON file
            
        Returns:
            bool: True if merging was successful, False otherwise
            str: Path to the output combined file if successful, None otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs("./combined_results", exist_ok=True)
            
            if not os.path.exists(nemo_file) or not os.path.exists(whisperx_file):
                logger.warning(f"Missing NeMo or WhisperX results for {audio_file_path}")
                return False, None
            
            # Read NeMo diarization results
            with open(nemo_file, 'r') as f:
                nemo_lines = f.readlines()
            
            # Parse the RTTM format from NeMo
            nemo_segments = []
            for line in nemo_lines:
                parts = line.strip().split()
                if len(parts) >= 8:
                    # RTTM format: SPEAKER file_id channel_id start_time duration NA NA speaker_id NA NA
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]
                    
                    nemo_segments.append({
                        'speaker': speaker_id,
                        'start': start_time,
                        'end': start_time + duration
                    })
            
            # Read WhisperX transcription results
            with open(whisperx_file, 'r') as f:
                whisperx_data = json.load(f)
            
            # Create a combined output
            base_name = os.path.basename(audio_file_path).split('.')[0]
            output_file = os.path.join("./combined_results", f"{base_name}_combined.txt")
            
            with open(output_file, 'w') as f:
                f.write(f"Combined diarization and transcription for {audio_file_path}\n")
                f.write("=" * 50 + "\n\n")
                
                # Use WhisperX segments with NeMo speaker IDs for better accuracy
                for segment in whisperx_data["segments"]:
                    start_time = segment["start"]
                    end_time = segment["end"]
                    text = segment["text"]
                    
                    # Find the most overlapping speaker from NeMo diarization
                    speaker_counts = {}
                    for nemo_seg in nemo_segments:
                        # Check for overlap
                        if max(start_time, nemo_seg['start']) < min(end_time, nemo_seg['end']):
                            overlap = min(end_time, nemo_seg['end']) - max(start_time, nemo_seg['start'])
                            speaker_counts[nemo_seg['speaker']] = speaker_counts.get(nemo_seg['speaker'], 0) + overlap
                    
                    # Get the speaker with maximum overlap
                    speaker = max(speaker_counts.items(), key=lambda x: x[1])[0] if speaker_counts else "UNKNOWN"
                    
                    # Format timestamps
                    start_time_str = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}.{int((start_time % 1) * 100):02d}"
                    end_time_str = f"{int(end_time // 60):02d}:{int(end_time % 60):02d}.{int((end_time % 1) * 100):02d}"
                    
                    f.write(f"[{start_time_str}] {speaker}: {text}\n")
            
            logger.info(f"Combined results saved to {output_file}")
            return True, output_file
            
        except Exception as e:
            logger.error(f"Error combining NeMo and WhisperX results: {str(e)}")
            return False, None