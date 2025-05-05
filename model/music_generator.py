"""
Music generation module for CustomTone.
Supports multiple models: YuE, MusicGen, and Magenta.
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

class MusicGenerator:
    """Base class for music generation models."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the music generator.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.model_type = self.config["model"]["type"]
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
        for dir_name in self.config["output"]["dirs"].values():
            os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
    
    def load_model(self) -> None:
        """Load the selected music generation model."""
        if self.model_type == "musicgen":
            self._load_musicgen()
        elif self.model_type == "yue":
            self._load_yue()
        elif self.model_type == "magenta":
            self._load_magenta()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Loaded {self.model_type} model on {self.device}")
    
    def _load_musicgen(self) -> None:
        """Load the MusicGen model."""
        try:
            from audiocraft.models import MusicGen
            
            model_size = self.config["model"]["size"]
            if model_size not in ["small", "medium", "large", "melody"]:
                logger.warning(f"Unknown model size: {model_size}, defaulting to 'medium'")
                model_size = "medium"
                
            self.model = MusicGen.get_pretrained(model_size, device=self.device)
            
            # Set generation parameters
            self.model.set_generation_params(
                duration=self.config["generation"]["max_duration"],
                temperature=self.config["generation"]["temperature"],
                top_k=self.config["generation"]["top_k"],
                top_p=self.config["generation"]["top_p"],
                cfg_coef=self.config["generation"]["cfg_scale"]
            )
        except ImportError:
            logger.error("Failed to import MusicGen. Install with: pip install -U audiocraft")
            raise
    
    def _load_yue(self) -> None:
        """Load the YuE model."""
        try:
            # This is a placeholder for YuE model loading
            # YuE is still in development and may require custom loading logic
            logger.warning("YuE model support is experimental")
            
            # Placeholder for YuE model loading
            # from yue.models import YuEModel
            # self.model = YuEModel.from_pretrained("multimodal-art-projection/YuE")
            
            # For now, we'll use a dummy model
            self.model = None
            logger.error("YuE model is not yet implemented. Please use MusicGen instead.")
            raise NotImplementedError("YuE model is not yet implemented")
        except ImportError:
            logger.error("Failed to import YuE. Check installation instructions.")
            raise
    
    def _load_magenta(self) -> None:
        """Load the Magenta model."""
        try:
            # Placeholder for Magenta model loading
            logger.warning("Magenta model support is experimental")
            
            # For now, we'll use a dummy model
            self.model = None
            logger.error("Magenta model is not yet fully implemented. Please use MusicGen instead.")
            raise NotImplementedError("Magenta model is not yet fully implemented")
        except ImportError:
            logger.error("Failed to import Magenta. Install with: pip install magenta")
            raise
    
    def generate(self, 
                prompt: str, 
                duration: Optional[float] = None,
                output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Generate music based on a text prompt.
        
        Args:
            prompt: Text description of the desired music.
            duration: Duration in seconds (overrides config if provided).
            output_path: Path to save the generated audio.
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self.model is None:
            self.load_model()
            
        if duration is None:
            duration = self.config["generation"]["max_duration"]
            
        sample_rate = self.config["generation"]["sample_rate"]
        
        logger.info(f"Generating music with prompt: '{prompt}'")
        
        if self.model_type == "musicgen":
            return self._generate_musicgen(prompt, duration, output_path, sample_rate)
        elif self.model_type == "yue":
            return self._generate_yue(prompt, duration, output_path, sample_rate)
        elif self.model_type == "magenta":
            return self._generate_magenta(prompt, duration, output_path, sample_rate)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_musicgen(self, 
                          prompt: str, 
                          duration: float,
                          output_path: Optional[str],
                          sample_rate: int) -> Tuple[np.ndarray, int]:
        """Generate music using MusicGen."""
        # Update duration if needed
        if duration != self.config["generation"]["max_duration"]:
            self.model.set_generation_params(duration=duration)
        
        # Generate audio
        audio_tensors = self.model.generate([prompt], progress=True)
        
        # Convert to numpy array
        audio_array = audio_tensors[0].cpu().numpy()
        
        # Save if output path is provided
        if output_path:
            self._save_audio(audio_array, output_path, sample_rate)
            
        return audio_array, sample_rate
    
    def _generate_yue(self, 
                     prompt: str, 
                     duration: float,
                     output_path: Optional[str],
                     sample_rate: int) -> Tuple[np.ndarray, int]:
        """Generate music using YuE."""
        # Placeholder for YuE generation
        raise NotImplementedError("YuE generation is not yet implemented")
    
    def _generate_magenta(self, 
                         prompt: str, 
                         duration: float,
                         output_path: Optional[str],
                         sample_rate: int) -> Tuple[np.ndarray, int]:
        """Generate music using Magenta."""
        # Placeholder for Magenta generation
        raise NotImplementedError("Magenta generation is not yet fully implemented")
    
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
