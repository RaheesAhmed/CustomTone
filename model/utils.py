"""
Utility functions for CustomTone.
"""

import os
import yaml
import numpy as np
import tempfile
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing the configuration.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def setup_directories(config: Dict) -> None:
    """Create output directories if they don't exist.
    
    Args:
        config: Configuration dictionary.
    """
    base_dir = config["output"]["base_dir"]
    for dir_name in config["output"]["dirs"].values():
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)

def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load audio from file.
    
    Args:
        audio_path: Path to the audio file.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    try:
        import soundfile as sf
        audio_array, sample_rate = sf.read(audio_path)
        return audio_array, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        raise

def save_audio(audio_array: np.ndarray, 
              output_path: str, 
              sample_rate: int) -> None:
    """Save audio array to file.
    
    Args:
        audio_array: Numpy array of audio data.
        output_path: Path to save the audio file.
        sample_rate: Sample rate of the audio.
    """
    try:
        import soundfile as sf
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio_array, sample_rate)
        logger.info(f"Saved audio to {output_path}")
    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        raise

def combine_stems(stem_paths: List[str], 
                 output_path: str,
                 gains: Optional[List[float]] = None) -> str:
    """Combine multiple audio stems into a single file.
    
    Args:
        stem_paths: List of paths to audio stems.
        output_path: Path to save the combined audio.
        gains: List of gain values for each stem (default: all 1.0).
        
    Returns:
        Path to the combined audio file.
    """
    try:
        from pydub import AudioSegment
        
        if gains is None:
            gains = [1.0] * len(stem_paths)
        
        if len(gains) != len(stem_paths):
            raise ValueError("Number of gains must match number of stems")
        
        # Initialize with silent audio
        combined = AudioSegment.silent(duration=0)
        max_length = 0
        
        # Load and combine stems
        for i, path in enumerate(stem_paths):
            if os.path.exists(path):
                stem = AudioSegment.from_file(path)
                
                # Apply gain
                stem = stem + (20 * np.log10(gains[i]))
                
                # Update max length
                max_length = max(max_length, len(stem))
                
                # Combine
                if len(combined) == 0:
                    combined = stem
                else:
                    # Ensure both segments are the same length
                    if len(combined) < len(stem):
                        combined = combined + AudioSegment.silent(duration=len(stem) - len(combined))
                    elif len(combined) > len(stem):
                        stem = stem + AudioSegment.silent(duration=len(combined) - len(stem))
                    
                    combined = combined.overlay(stem)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export
        combined.export(output_path, format=os.path.splitext(output_path)[1][1:])
        logger.info(f"Saved combined audio to {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Error combining stems: {e}")
        raise

def convert_audio_format(input_path: str, 
                        output_format: str = "wav",
                        output_path: Optional[str] = None) -> str:
    """Convert audio to a different format.
    
    Args:
        input_path: Path to the input audio file.
        output_format: Output format (e.g., "wav", "mp3", "flac").
        output_path: Path to save the converted audio (default: same as input with new extension).
        
    Returns:
        Path to the converted audio file.
    """
    try:
        from pydub import AudioSegment
        
        # Determine output path
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + "." + output_format
        
        # Load audio
        audio = AudioSegment.from_file(input_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export
        audio.export(output_path, format=output_format)
        logger.info(f"Converted {input_path} to {output_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        raise

def normalize_audio(audio_array: np.ndarray, 
                   target_db: float = -3.0) -> np.ndarray:
    """Normalize audio to a target dB level.
    
    Args:
        audio_array: Numpy array of audio data.
        target_db: Target dB level (default: -3.0).
        
    Returns:
        Normalized audio array.
    """
    try:
        # Calculate current dB level
        current_db = 20 * np.log10(np.max(np.abs(audio_array)) + 1e-8)
        
        # Calculate gain
        gain = 10 ** ((target_db - current_db) / 20)
        
        # Apply gain
        normalized = audio_array * gain
        
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing audio: {e}")
        raise

def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file.
        
    Returns:
        Duration in seconds.
    """
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        raise

def get_audio_info(audio_path: str) -> Dict:
    """Get information about an audio file.
    
    Args:
        audio_path: Path to the audio file.
        
    Returns:
        Dictionary containing audio information.
    """
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "file_size": os.path.getsize(audio_path)
        }
    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        raise

def create_spectrogram(audio_array: np.ndarray, 
                      sample_rate: int,
                      output_path: Optional[str] = None) -> np.ndarray:
    """Create a spectrogram from audio data.
    
    Args:
        audio_array: Numpy array of audio data.
        sample_rate: Sample rate of the audio.
        output_path: Path to save the spectrogram image (optional).
        
    Returns:
        Numpy array of the spectrogram.
    """
    try:
        import librosa
        import matplotlib.pyplot as plt
        
        # Convert to mono if stereo
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Create spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio_array)), ref=np.max
        )
        
        # Save if output path is provided
        if output_path:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                D, sr=sample_rate, x_axis='time', y_axis='log'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved spectrogram to {output_path}")
        
        return D
    except Exception as e:
        logger.error(f"Error creating spectrogram: {e}")
        raise
