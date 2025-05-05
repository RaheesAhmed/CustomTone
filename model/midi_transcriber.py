"""
MIDI transcription module for CustomTone.
Converts audio stems to MIDI for detailed editing.
"""

import os
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MidiTranscriber:
    """Class for transcribing audio to MIDI."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the MIDI transcriber.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.transcriber_type = self.config["transcription"]["model"]
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
    
    def _setup_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        base_dir = self.config["output"]["base_dir"]
        midi_dir = os.path.join(base_dir, self.config["output"]["dirs"]["midi"])
        os.makedirs(midi_dir, exist_ok=True)
    
    def load_model(self) -> None:
        """Load the selected MIDI transcription model."""
        if self.transcriber_type == "basic_pitch":
            self._load_basic_pitch()
        elif self.transcriber_type == "onsets_frames":
            self._load_onsets_frames()
        else:
            raise ValueError(f"Unsupported transcriber type: {self.transcriber_type}")
        
        logger.info(f"Loaded {self.transcriber_type} model")
    
    def _load_basic_pitch(self) -> None:
        """Load the BasicPitch model."""
        try:
            from basic_pitch.inference import Transcriber
            
            self.model = Transcriber()
            logger.info("BasicPitch model loaded")
        except ImportError:
            logger.error("Failed to import BasicPitch. Install with: pip install basic-pitch")
            raise
    
    def _load_onsets_frames(self) -> None:
        """Load the Onsets and Frames model."""
        try:
            # Placeholder for Onsets and Frames model loading
            logger.warning("Onsets and Frames model support is experimental")
            
            # For now, we'll use a dummy model
            self.model = None
            logger.error("Onsets and Frames model is not yet implemented. Please use BasicPitch instead.")
            raise NotImplementedError("Onsets and Frames model is not yet implemented")
        except ImportError:
            logger.error("Failed to import Onsets and Frames. Install with: pip install magenta")
            raise
    
    def transcribe(self, 
                  audio: Union[str, np.ndarray], 
                  sample_rate: Optional[int] = None,
                  output_dir: Optional[str] = None,
                  stem_name: Optional[str] = None) -> str:
        """Transcribe audio to MIDI.
        
        Args:
            audio: Path to audio file or numpy array of audio data.
            sample_rate: Sample rate of the audio (required if audio is numpy array).
            output_dir: Directory to save the MIDI file.
            stem_name: Name of the stem (for naming the output file).
            
        Returns:
            Path to the generated MIDI file.
        """
        if self.model is None:
            self.load_model()
            
        if output_dir is None:
            base_dir = self.config["output"]["base_dir"]
            output_dir = os.path.join(base_dir, self.config["output"]["dirs"]["midi"])
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio if path is provided
        if isinstance(audio, str):
            audio_path = audio
            if stem_name is None:
                stem_name = os.path.splitext(os.path.basename(audio_path))[0]
        else:
            # Save audio to temporary file if it's a numpy array
            if stem_name is None:
                stem_name = "unnamed_stem"
                
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_path = temp_file.name
                self._save_audio(audio, audio_path, sample_rate)
        
        logger.info(f"Transcribing audio to MIDI using {self.transcriber_type}")
        
        if self.transcriber_type == "basic_pitch":
            return self._transcribe_basic_pitch(audio_path, output_dir, stem_name)
        elif self.transcriber_type == "onsets_frames":
            return self._transcribe_onsets_frames(audio_path, output_dir, stem_name)
        else:
            raise ValueError(f"Unsupported transcriber type: {self.transcriber_type}")
    
    def _transcribe_basic_pitch(self, 
                              audio_path: str, 
                              output_dir: str,
                              stem_name: str) -> str:
        """Transcribe audio using BasicPitch."""
        # Set output MIDI path
        midi_path = os.path.join(output_dir, f"{stem_name}.{self.config['output']['formats']['midi']}")
        
        # Get transcription parameters
        min_note_confidence = self.config["transcription"]["min_note_confidence"]
        min_note_duration = self.config["transcription"]["min_note_duration"]
        
        # Transcribe
        model_output, midi_data, note_events = self.model.transcribe(
            audio_path,
            onset_threshold=min_note_confidence,
            frame_threshold=min_note_confidence,
            minimum_note_length=min_note_duration
        )
        
        # Save MIDI
        if self.config["transcription"]["save_midi"]:
            midi_data.write(midi_path)
            logger.info(f"Saved MIDI to {midi_path}")
        
        # Save note events if requested
        if self.config["transcription"]["save_note_events"]:
            import json
            note_events_path = os.path.join(output_dir, f"{stem_name}_notes.json")
            
            # Convert note events to serializable format
            serializable_notes = []
            for note in note_events:
                serializable_note = {
                    "start_time": float(note.start_time),
                    "end_time": float(note.end_time),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "confidence": float(note.confidence)
                }
                serializable_notes.append(serializable_note)
                
            with open(note_events_path, "w") as f:
                json.dump(serializable_notes, f, indent=2)
            
            logger.info(f"Saved note events to {note_events_path}")
        
        return midi_path
    
    def _transcribe_onsets_frames(self, 
                                audio_path: str, 
                                output_dir: str,
                                stem_name: str) -> str:
        """Transcribe audio using Onsets and Frames."""
        # Placeholder for Onsets and Frames transcription
        raise NotImplementedError("Onsets and Frames transcription is not yet implemented")
    
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
