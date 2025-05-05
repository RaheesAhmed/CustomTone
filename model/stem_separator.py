"""
Stem separation module for CustomTone.
Supports Demucs and Spleeter for separating audio into instrument stems.
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StemSeparator:
    """Class for separating audio into instrument stems."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the stem separator.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.separator_type = self.config["separation"]["model"]
        self.model = None
        self._setup_output_dirs()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
            
    def _get_device(self) -> torch.device:
        """Get the appropriate device (CPU or CUDA)."""
        if self.config["system"]["device"] == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config["system"]["device"])
    
    def _setup_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        base_dir = self.config["output"]["base_dir"]
        stems_dir = os.path.join(base_dir, self.config["output"]["dirs"]["stems"])
        os.makedirs(stems_dir, exist_ok=True)
    
    def load_model(self) -> None:
        """Load the selected stem separation model."""
        if self.separator_type == "demucs":
            self._load_demucs()
        elif self.separator_type == "spleeter":
            self._load_spleeter()
        else:
            raise ValueError(f"Unsupported separator type: {self.separator_type}")
        
        logger.info(f"Loaded {self.separator_type} model on {self.device}")
    
    def _load_demucs(self) -> None:
        """Load the Demucs model."""
        try:
            from demucs.pretrained import get_model
            
            model_version = self.config["separation"]["demucs_version"]
            self.model = get_model(model_version)
            self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
        except ImportError:
            logger.error("Failed to import Demucs. Install with: pip install demucs")
            raise
    
    def _load_spleeter(self) -> None:
        """Load the Spleeter model."""
        try:
            # Spleeter is typically used via command line
            # We'll use a flag to indicate it's loaded
            self.model = "spleeter"
            
            # Check if spleeter is installed
            import spleeter
            logger.info("Spleeter is available")
        except ImportError:
            logger.error("Failed to import Spleeter. Install with: pip install spleeter")
            raise
    
    def separate(self, 
                audio: Union[str, np.ndarray], 
                sample_rate: Optional[int] = None,
                output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Separate audio into stems.
        
        Args:
            audio: Path to audio file or numpy array of audio data.
            sample_rate: Sample rate of the audio (required if audio is numpy array).
            output_dir: Directory to save the separated stems.
            
        Returns:
            Dictionary mapping stem names to numpy arrays of audio data.
        """
        if self.model is None:
            self.load_model()
            
        if output_dir is None:
            base_dir = self.config["output"]["base_dir"]
            output_dir = os.path.join(base_dir, self.config["output"]["dirs"]["stems"])
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio if path is provided
        if isinstance(audio, str):
            audio_path = audio
            audio_array, sample_rate = self._load_audio(audio_path)
        else:
            audio_array = audio
            if sample_rate is None:
                raise ValueError("Sample rate must be provided when audio is a numpy array")
        
        logger.info(f"Separating audio into stems using {self.separator_type}")
        
        if self.separator_type == "demucs":
            return self._separate_demucs(audio_array, sample_rate, output_dir)
        elif self.separator_type == "spleeter":
            return self._separate_spleeter(audio_array, sample_rate, output_dir)
        else:
            raise ValueError(f"Unsupported separator type: {self.separator_type}")
    
    def _separate_demucs(self, 
                        audio_array: np.ndarray, 
                        sample_rate: int,
                        output_dir: str) -> Dict[str, np.ndarray]:
        """Separate audio using Demucs."""
        # Convert to torch tensor
        if audio_array.ndim == 1:
            # Convert mono to stereo
            audio_array = np.stack([audio_array, audio_array])
        elif audio_array.ndim == 2 and audio_array.shape[0] == 1:
            # Convert mono to stereo
            audio_array = np.stack([audio_array[0], audio_array[0]])
        elif audio_array.ndim == 2 and audio_array.shape[1] == 1:
            # Convert mono to stereo
            audio_array = np.stack([audio_array[:, 0], audio_array[:, 0]])
        elif audio_array.ndim == 2 and audio_array.shape[0] == 2:
            # Already stereo, do nothing
            pass
        elif audio_array.ndim == 2:
            # Assume channels are in the second dimension
            audio_array = audio_array.T
        else:
            raise ValueError(f"Unsupported audio shape: {audio_array.shape}")
        
        # Ensure we have a proper torch tensor
        audio_tensor = torch.tensor(audio_array, device=self.device)
        
        # Normalize if needed
        if audio_tensor.abs().max() > 1:
            audio_tensor = audio_tensor / audio_tensor.abs().max()
        
        # Separate
        with torch.no_grad():
            stems = self.model(audio_tensor.unsqueeze(0))
            
        # Get stem names
        stem_names = self.config["separation"]["stem_names"]
        
        # Convert back to numpy and create dictionary
        stems_dict = {}
        for i, name in enumerate(stem_names):
            stem_array = stems[0, i].cpu().numpy()
            stems_dict[name] = stem_array
            
            # Save stem
            stem_path = os.path.join(output_dir, f"{name}.wav")
            self._save_audio(stem_array, stem_path, sample_rate)
        
        return stems_dict
    
    def _separate_spleeter(self, 
                          audio_array: np.ndarray, 
                          sample_rate: int,
                          output_dir: str) -> Dict[str, np.ndarray]:
        """Separate audio using Spleeter."""
        # Save audio to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            self._save_audio(audio_array, temp_path, sample_rate)
        
        # Determine number of stems
        num_stems = self.config["separation"]["num_stems"]
        stem_config = f"spleeter:{num_stems}stems"
        
        # Run spleeter
        import subprocess
        cmd = [
            "spleeter", "separate",
            "-i", temp_path,
            "-p", stem_config,
            "-o", output_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Load separated stems
            stems_dict = {}
            stem_names = self.config["separation"]["stem_names"]
            
            # Spleeter creates a subdirectory with the name of the input file
            spleeter_output_dir = os.path.join(output_dir, os.path.basename(temp_path).split(".")[0])
            
            for name in stem_names:
                stem_path = os.path.join(spleeter_output_dir, f"{name}.wav")
                if os.path.exists(stem_path):
                    stem_array, _ = self._load_audio(stem_path)
                    stems_dict[name] = stem_array
            
            return stems_dict
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        try:
            import soundfile as sf
            audio_array, sample_rate = sf.read(audio_path)
            return audio_array, sample_rate
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def _save_audio(self, 
                   audio_array: np.ndarray, 
                   output_path: str, 
                   sample_rate: int) -> None:
        """Save audio array to file."""
        try:
            import soundfile as sf
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio_array, sample_rate)
            logger.info(f"Saved audio to {output_path}")
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise
