"""
CustomTone: Customizable Music Generation System
Gradio interface for generating, separating, and customizing music.
"""

import os
import yaml
import numpy as np
import gradio as gr
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import CustomTone modules
from model.music_generator import MusicGenerator
from model.stem_separator import StemSeparator
from model.midi_transcriber import MidiTranscriber
import model.utils as utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = utils.load_config()

# Create output directories
utils.setup_directories(config)

class CustomToneApp:
    """Gradio interface for CustomTone."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the CustomTone app.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = utils.load_config(config_path)
        self.music_generator = MusicGenerator(config_path)
        self.stem_separator = StemSeparator(config_path)
        self.midi_transcriber = MidiTranscriber(config_path)
        
        # Set up output paths
        self.base_dir = self.config["output"]["base_dir"]
        self.full_mix_dir = os.path.join(self.base_dir, self.config["output"]["dirs"]["full_mix"])
        self.stems_dir = os.path.join(self.base_dir, self.config["output"]["dirs"]["stems"])
        self.midi_dir = os.path.join(self.base_dir, self.config["output"]["dirs"]["midi"])
        self.custom_mix_dir = os.path.join(self.base_dir, self.config["output"]["dirs"]["custom_mix"])
        
        # Create directories
        for directory in [self.full_mix_dir, self.stems_dir, self.midi_dir, self.custom_mix_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize state
        self.current_full_mix = None
        self.current_stems = {}
        self.current_midi_files = {}
    
    def generate_music(self, 
                      prompt: str, 
                      duration: float) -> Tuple[str, Dict[str, str]]:
        """Generate music based on a text prompt.
        
        Args:
            prompt: Text description of the desired music.
            duration: Duration in seconds.
            
        Returns:
            Tuple of (full_mix_path, stem_paths).
        """
        logger.info(f"Generating music with prompt: '{prompt}', duration: {duration}s")
        
        # Generate music
        output_path = os.path.join(self.full_mix_dir, "full_mix.wav")
        audio_array, sample_rate = self.music_generator.generate(
            prompt=prompt,
            duration=duration,
            output_path=output_path
        )
        
        # Store current full mix
        self.current_full_mix = output_path
        
        # Separate stems
        stem_dict = self.stem_separator.separate(
            audio=output_path,
            output_dir=self.stems_dir
        )
        
        # Store current stems
        self.current_stems = {}
        for stem_name, stem_array in stem_dict.items():
            stem_path = os.path.join(self.stems_dir, f"{stem_name}.wav")
            self.current_stems[stem_name] = stem_path
        
        # Generate MIDI for each stem
        self.current_midi_files = {}
        for stem_name, stem_path in self.current_stems.items():
            try:
                midi_path = self.midi_transcriber.transcribe(
                    audio=stem_path,
                    output_dir=self.midi_dir,
                    stem_name=stem_name
                )
                self.current_midi_files[stem_name] = midi_path
            except Exception as e:
                logger.error(f"Error transcribing {stem_name}: {e}")
        
        return output_path, self.current_stems
    
    def customize_mix(self, 
                     selected_stems: List[str],
                     stem_gains: List[float]) -> str:
        """Create a custom mix from selected stems.
        
        Args:
            selected_stems: List of stem names to include.
            stem_gains: List of gain values for each stem.
            
        Returns:
            Path to the custom mix.
        """
        if not self.current_stems:
            raise ValueError("No stems available. Generate music first.")
        
        # Filter stems and gains
        stem_paths = [self.current_stems[stem] for stem in selected_stems if stem in self.current_stems]
        gains = stem_gains[:len(stem_paths)]
        
        # Create custom mix
        output_path = os.path.join(self.custom_mix_dir, "custom_mix.wav")
        utils.combine_stems(stem_paths, output_path, gains)
        
        return output_path
    
    def create_gradio_interface(self) -> gr.Blocks:
        """Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface.
        """
        with gr.Blocks(title="CustomTone: Customizable Music Generation") as interface:
            gr.Markdown("# 🎵 CustomTone: Customizable Music Generation")
            gr.Markdown("Generate music with full control over individual instruments")
            
            with gr.Tab("Generate Music"):
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="Music Description or Lyrics",
                            placeholder="Enter a description of the music you want to generate...",
                            lines=3
                        )
                        duration = gr.Slider(
                            label="Duration (seconds)",
                            minimum=5,
                            maximum=60,
                            value=15,
                            step=5
                        )
                        generate_btn = gr.Button("Generate Music", variant="primary")
                    
                    with gr.Column():
                        output_audio = gr.Audio(label="Generated Music")
                
                with gr.Row():
                    status = gr.Textbox(label="Status", value="Ready")
            
            with gr.Tab("Customize Stems"):
                with gr.Row():
                    with gr.Column():
                        available_stems = gr.CheckboxGroup(
                            label="Select Stems to Include",
                            choices=["vocals", "drums", "bass", "other"],
                            value=["vocals", "drums", "bass", "other"]
                        )
                        
                        with gr.Row():
                            vocals_gain = gr.Slider(label="Vocals Gain", minimum=0, maximum=2, value=1, step=0.1)
                            drums_gain = gr.Slider(label="Drums Gain", minimum=0, maximum=2, value=1, step=0.1)
                        
                        with gr.Row():
                            bass_gain = gr.Slider(label="Bass Gain", minimum=0, maximum=2, value=1, step=0.1)
                            other_gain = gr.Slider(label="Other Gain", minimum=0, maximum=2, value=1, step=0.1)
                        
                        customize_btn = gr.Button("Create Custom Mix", variant="primary")
                    
                    with gr.Column():
                        custom_mix_audio = gr.Audio(label="Custom Mix")
                        
                        with gr.Accordion("Individual Stems", open=False):
                            vocals_audio = gr.Audio(label="Vocals")
                            drums_audio = gr.Audio(label="Drums")
                            bass_audio = gr.Audio(label="Bass")
                            other_audio = gr.Audio(label="Other")
            
            with gr.Tab("Download MIDI"):
                with gr.Row():
                    with gr.Column():
                        midi_stem_select = gr.Radio(
                            label="Select Stem for MIDI",
                            choices=["vocals", "drums", "bass", "other"],
                            value="vocals"
                        )
                        download_midi_btn = gr.Button("Download MIDI", variant="primary")
                    
                    with gr.Column():
                        midi_file = gr.File(label="MIDI File")
                        midi_info = gr.Textbox(label="MIDI Info", lines=3)
            
            # Define functions for the interface
            def generate_music_handler(prompt, duration):
                try:
                    full_mix, stems = self.generate_music(prompt, duration)
                    
                    # Prepare individual stems for display
                    vocals = stems.get("vocals", None)
                    drums = stems.get("drums", None)
                    bass = stems.get("bass", None)
                    other = stems.get("other", None)
                    
                    return {
                        output_audio: full_mix,
                        vocals_audio: vocals,
                        drums_audio: drums,
                        bass_audio: bass,
                        other_audio: other,
                        status: f"Generated music with prompt: '{prompt}'"
                    }
                except Exception as e:
                    logger.error(f"Error generating music: {e}")
                    return {status: f"Error: {str(e)}"}
            
            def customize_mix_handler(selected_stems, vocals_gain, drums_gain, bass_gain, other_gain):
                try:
                    # Map gains to selected stems
                    gain_map = {
                        "vocals": vocals_gain,
                        "drums": drums_gain,
                        "bass": bass_gain,
                        "other": other_gain
                    }
                    
                    gains = [gain_map[stem] for stem in selected_stems]
                    
                    custom_mix = self.customize_mix(selected_stems, gains)
                    
                    return {
                        custom_mix_audio: custom_mix,
                        status: f"Created custom mix with stems: {', '.join(selected_stems)}"
                    }
                except Exception as e:
                    logger.error(f"Error customizing mix: {e}")
                    return {status: f"Error: {str(e)}"}
            
            def download_midi_handler(stem_name):
                try:
                    if not self.current_midi_files:
                        return {
                            midi_info: "No MIDI files available. Generate music first.",
                            midi_file: None
                        }
                    
                    if stem_name not in self.current_midi_files:
                        return {
                            midi_info: f"No MIDI file available for {stem_name}.",
                            midi_file: None
                        }
                    
                    midi_path = self.current_midi_files[stem_name]
                    
                    return {
                        midi_file: midi_path,
                        midi_info: f"MIDI file for {stem_name} ready for download."
                    }
                except Exception as e:
                    logger.error(f"Error downloading MIDI: {e}")
                    return {
                        midi_info: f"Error: {str(e)}",
                        midi_file: None
                    }
            
            # Connect functions to buttons
            generate_btn.click(
                generate_music_handler,
                inputs=[prompt, duration],
                outputs=[output_audio, vocals_audio, drums_audio, bass_audio, other_audio, status]
            )
            
            customize_btn.click(
                customize_mix_handler,
                inputs=[available_stems, vocals_gain, drums_gain, bass_gain, other_gain],
                outputs=[custom_mix_audio, status]
            )
            
            download_midi_btn.click(
                download_midi_handler,
                inputs=[midi_stem_select],
                outputs=[midi_file, midi_info]
            )
        
        return interface
    
    def launch(self, share: bool = False) -> None:
        """Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link.
        """
        interface = self.create_gradio_interface()
        
        # Get Gradio settings from config
        port = self.config["gradio"]["port"]
        theme = self.config["gradio"]["theme"]
        share_config = self.config["gradio"]["share"] if not share else share
        
        # Launch the interface
        interface.launch(
            server_port=port,
            share=share_config,
            server_name="0.0.0.0"  # Allow external connections
        )

if __name__ == "__main__":
    app = CustomToneApp()
    app.launch()
