#!/usr/bin/env python3
"""
Audio File Splitter

This script processes audio files in a directory, splitting files longer than a specified
duration into smaller segments and moving the original files to an "originals" directory.

Usage:
  python audio_splitter.py [input_dir] [--originals DIR] [--duration MINUTES] [--recursive]
"""

import os
import sys
import shlex
import shutil
import argparse
import importlib.util
from pathlib import Path

MAX_LENGTH_MINS = 15

def check_dependencies():
    """Check and install required dependencies if not already installed."""
    # Check for ffmpeg-python
    if importlib.util.find_spec("ffmpeg") is None:
        print("ffmpeg-python is missing")
        return False
    
    # Check if ffmpeg command-line tool is installed
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("ffmpeg command-line tool is available.")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: ffmpeg command-line tool is not installed or not in PATH.")
        print("Installation instructions: https://ffmpeg.org/download.html")
        return False
    
    return True

def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds using ffmpeg.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds or None if an error occurs
    """
    import ffmpeg
    
    try:
        probe = ffmpeg.probe(file_path)
        # Get duration from the format information
        duration = float(probe['format']['duration'])
        return duration
    except ffmpeg.Error as e:
        print(f"Error probing {file_path}: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        return None

def split_audio_file(file_path, output_dir, originals_dir, segment_duration=MAX_LENGTH_MINS*60):
    """
    Split audio file into segments of specified duration with 30-second overlap.
    
    Args:
        file_path: Path to the audio file
        output_dir: Directory to save the split files
        originals_dir: Directory to move the original file
        segment_duration: Duration of each segment in seconds (default: 15 minutes)
        
    Returns:
        Boolean indicating success or failure
    """
    import ffmpeg
    
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    originals_dir = Path(originals_dir)
    
    # Create directories if they don't exist
    output_dir.mkdir(exist_ok=True)
    originals_dir.mkdir(exist_ok=True)
    
    # Get file duration
    duration = get_audio_duration(file_path)
    if duration is None:
        print(f"Skipping {file_path} due to error getting duration.")
        return False
    
    print(f"Processing: {file_path.name} (Duration: {duration/60:.2f} minutes)")
    
    # If duration is less than segment_duration, no need to split
    if duration <= segment_duration:
        print(f"File {file_path.name} is shorter than {segment_duration/60} minutes. No splitting needed.")
        return True
    
    # Define overlap duration (30 seconds)
    overlap_duration = 30
    
    # Calculate number of segments needed
    # We need to recalculate this because with overlap, we might need more segments
    effective_segment_duration = segment_duration - overlap_duration
    num_segments = int((duration - overlap_duration) / effective_segment_duration) + 1
    
    # Get file name without extension and the extension
    file_stem = file_path.stem
    file_ext = file_path.suffix
    
    # Track if all segments were created successfully
    all_segments_success = True
    
    # Process each segment
    for i in range(num_segments):
        # Calculate segment start time with overlap
        # First segment starts at 0, subsequent segments start with overlap
        segment_start = max(0, i * effective_segment_duration)
        
        # For the last segment, make sure we don't exceed the file duration
        current_segment_duration = min(segment_duration, duration - segment_start)
        
        output_filename = f"{file_stem}_{i+1:02d}{file_ext}"
        output_path = Path(output_dir / output_filename)
        
        print(f"Creating segment {i+1}/{num_segments}: {output_filename} (Start: {segment_start}s, Duration: {current_segment_duration}s)")
        
        try:
            # Convert paths to strings for better handling with spaces
            input_path_str = str(file_path.resolve())
            output_path_str = str(output_path.resolve())
            
            # Use ffmpeg to extract the segment
            (
                ffmpeg
                .input(input_path_str, ss=segment_start, t=current_segment_duration, y=None)
                .output(output_path_str, acodec='copy', loglevel='error', hide_banner=None)
                .run(capture_stdout=True, capture_stderr=True)
            )
            print(f"Created: {output_filename}")
            
            # Verify the segment was created successfully
            if not output_path.exists():
                print(f"Error: Segment file {output_filename} was not created.")
                all_segments_success = False
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode('utf-8') if (hasattr(e, 'stderr') and e.stderr) else str(e)
            print(f"Error creating segment {i+1}: {error_msg}")
            # Check if error is related to file paths
            if "No such file or directory" in error_msg or "Invalid argument" in error_msg:
                print(f"This may be due to spaces in the file path. Input path: '{input_path_str}', Output path: '{output_path_str}'")
            all_segments_success = False
    
    # Only move the original file if all segments were created successfully
    if all_segments_success:
        # Move original file to originals directory
        dest_path = originals_dir / file_path.name
        shutil.move(str(file_path), str(dest_path))
        print(f"Moved original file {file_path.name} to {originals_dir}")
        return True
    else:
        print(f"Warning: Not all segments were created successfully for {file_path.name}.")
        print("Original file was not moved to prevent data loss.")
        return False

def process_directory(input_dir, originals_dir=None, segment_duration=MAX_LENGTH_MINS*60):
    """
    Process all audio files in the specified directory.
    
    Args:
        input_dir: Directory containing audio files
        originals_dir: Directory to move original files (defaults to 'originals' subdirectory)
        segment_duration: Duration of each segment in seconds
    """
    input_dir = Path(input_dir)
    
    # If originals_dir is not specified, create a subdirectory called 'originals'
    originals_dir = Path(originals_dir) if originals_dir else input_dir / 'originals'
    
    # Create originals directory if it doesn't exist
    originals_dir.mkdir(exist_ok=True)
    
    # Audio file extensions to process
    audio_extensions = ['.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.amr', '.wma', '.ac3', '.mpa']
    
    # Find all audio files in the directory
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio file(s) in {input_dir}")
    
    # Process each audio file
    successful = 0
    failed = 0
    
    for file_path in audio_files:
        # Skip files in the originals directory
        if originals_dir in file_path.parents:
            continue
        
        if split_audio_file(file_path, input_dir, originals_dir, segment_duration):
            successful += 1
        else:
            failed += 1
    
    print("\nProcessing summary:")
    print(f"- Successfully processed: {successful} file(s)")
    print(f"- Failed to process: {failed} file(s)")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Split audio files longer than a specified duration into smaller segments.'
    )
    parser.add_argument(
        'input_dir', 
        nargs='?', 
        default=os.getcwd(),
        help='Directory containing audio files (default: current directory)'
    )
    parser.add_argument(
        '--originals', '-o',
        help='Directory to move original files (default: "originals" subdirectory in input directory)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=15,
        help='Duration of each segment in minutes (default: 15)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process subdirectories recursively'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("Error: Missing required dependencies. Please install ffmpeg.")
        return
    
    input_dir = args.input_dir
    originals_dir = args.originals
    segment_duration = args.duration * 60  # Convert minutes to seconds
    
    print(f"Processing audio files in: {input_dir}")
    print(f"Moving originals to: {originals_dir or os.path.join(input_dir, 'originals')}")
    print(f"Segment duration: {args.duration} minutes")
    
    # Process the directory
    if args.recursive:
        # Process the main directory and all subdirectories
        for root, _, _ in os.walk(input_dir):
            # Skip the originals directory
            if originals_dir and Path(root) == Path(originals_dir):
                continue
            
            print(f"\nProcessing directory: {root}")
            process_directory(root, originals_dir, segment_duration)
    else:
        # Process only the specified directory
        process_directory(input_dir, originals_dir, segment_duration)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
