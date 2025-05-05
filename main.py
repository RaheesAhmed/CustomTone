"""
CustomTone: Customizable Music Generation System
Command-line interface for generating, separating, and customizing music.
"""

import os
import argparse
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Union

# Import CustomTone modules
from model.music_generator import MusicGenerator
from model.stem_separator import StemSeparator
from model.midi_transcriber import MidiTranscriber
import model.utils as utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CustomTone: Customizable Music Generation")
    
    # Main command
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate music from text prompt")
    generate_parser.add_argument("--prompt", "-p", type=str, required=True, help="Text description of the desired music")
    generate_parser.add_argument("--duration", "-d", type=float, default=15.0, help="Duration in seconds")
    generate_parser.add_argument("--output", "-o", type=str, default=None, help="Output path for the generated audio")
    generate_parser.add_argument("--model", "-m", type=str, choices=["musicgen", "yue", "magenta"], default=None, help="Model to use")
    generate_parser.add_argument("--no-separate", action="store_true", help="Skip stem separation")
    generate_parser.add_argument("--no-midi", action="store_true", help="Skip MIDI transcription")
    
    # Separate command
    separate_parser = subparsers.add_parser("separate", help="Separate audio into stems")
    separate_parser.add_argument("--input", "-i", type=str, required=True, help="Input audio file")
    separate_parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for stems")
    separate_parser.add_argument("--model", "-m", type=str, choices=["demucs", "spleeter"], default=None, help="Model to use")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio to MIDI")
    transcribe_parser.add_argument("--input", "-i", type=str, required=True, help="Input audio file")
    transcribe_parser.add_argument("--output", "-o", type=str, default=None, help="Output MIDI file")
    transcribe_parser.add_argument("--model", "-m", type=str, choices=["basic_pitch", "onsets_frames"], default=None, help="Model to use")
    
    # Customize command
    customize_parser = subparsers.add_parser("customize", help="Create custom mix from stems")
    customize_parser.add_argument("--stems", "-s", type=str, nargs="+", required=True, help="Stem files to include")
    customize_parser.add_argument("--gains", "-g", type=float, nargs="+", default=None, help="Gain for each stem")
    customize_parser.add_argument("--output", "-o", type=str, default=None, help="Output path for the custom mix")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch the Gradio UI")
    ui_parser.add_argument("--share", action="store_true", help="Create a public link")
    
    # Config
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to configuration file")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = utils.load_config(args.config)
    
    # Create output directories
    utils.setup_directories(config)
    
    if args.command == "generate":
        # Override model if specified
        if args.model:
            config["model"]["type"] = args.model
        
        # Initialize music generator
        music_generator = MusicGenerator(args.config)
        
        # Generate music
        if args.output:
            output_path = args.output
        else:
            base_dir = config["output"]["base_dir"]
            full_mix_dir = os.path.join(base_dir, config["output"]["dirs"]["full_mix"])
            output_path = os.path.join(full_mix_dir, "full_mix.wav")
        
        audio_array, sample_rate = music_generator.generate(
            prompt=args.prompt,
            duration=args.duration,
            output_path=output_path
        )
        
        logger.info(f"Generated music saved to {output_path}")
        
        # Separate stems if requested
        if not args.no_separate:
            # Initialize stem separator
            stem_separator = StemSeparator(args.config)
            
            # Separate stems
            base_dir = config["output"]["base_dir"]
            stems_dir = os.path.join(base_dir, config["output"]["dirs"]["stems"])
            
            stem_dict = stem_separator.separate(
                audio=output_path,
                output_dir=stems_dir
            )
            
            logger.info(f"Separated stems saved to {stems_dir}")
            
            # Transcribe to MIDI if requested
            if not args.no_midi:
                # Initialize MIDI transcriber
                midi_transcriber = MidiTranscriber(args.config)
                
                # Transcribe each stem
                base_dir = config["output"]["base_dir"]
                midi_dir = os.path.join(base_dir, config["output"]["dirs"]["midi"])
                
                for stem_name, stem_array in stem_dict.items():
                    stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
                    
                    try:
                        midi_path = midi_transcriber.transcribe(
                            audio=stem_path,
                            output_dir=midi_dir,
                            stem_name=stem_name
                        )
                        logger.info(f"Transcribed {stem_name} to MIDI: {midi_path}")
                    except Exception as e:
                        logger.error(f"Error transcribing {stem_name}: {e}")
    
    elif args.command == "separate":
        # Override model if specified
        if args.model:
            config["separation"]["model"] = args.model
        
        # Initialize stem separator
        stem_separator = StemSeparator(args.config)
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            base_dir = config["output"]["base_dir"]
            output_dir = os.path.join(base_dir, config["output"]["dirs"]["stems"])
        
        # Separate stems
        stem_dict = stem_separator.separate(
            audio=args.input,
            output_dir=output_dir
        )
        
        logger.info(f"Separated stems saved to {output_dir}")
    
    elif args.command == "transcribe":
        # Override model if specified
        if args.model:
            config["transcription"]["model"] = args.model
        
        # Initialize MIDI transcriber
        midi_transcriber = MidiTranscriber(args.config)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_dir = config["output"]["base_dir"]
            midi_dir = os.path.join(base_dir, config["output"]["dirs"]["midi"])
            stem_name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(midi_dir, f"{stem_name}.mid")
        
        # Transcribe
        midi_path = midi_transcriber.transcribe(
            audio=args.input,
            output_dir=os.path.dirname(output_path),
            stem_name=os.path.splitext(os.path.basename(output_path))[0]
        )
        
        logger.info(f"Transcribed audio to MIDI: {midi_path}")
    
    elif args.command == "customize":
        # Check if all stems exist
        for stem_path in args.stems:
            if not os.path.exists(stem_path):
                logger.error(f"Stem file not found: {stem_path}")
                return
        
        # Determine gains
        if args.gains:
            if len(args.gains) != len(args.stems):
                logger.warning(f"Number of gains ({len(args.gains)}) does not match number of stems ({len(args.stems)})")
                gains = args.gains[:len(args.stems)] + [1.0] * (len(args.stems) - len(args.gains))
            else:
                gains = args.gains
        else:
            gains = [1.0] * len(args.stems)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_dir = config["output"]["base_dir"]
            custom_mix_dir = os.path.join(base_dir, config["output"]["dirs"]["custom_mix"])
            output_path = os.path.join(custom_mix_dir, "custom_mix.wav")
        
        # Combine stems
        utils.combine_stems(args.stems, output_path, gains)
        
        logger.info(f"Custom mix saved to {output_path}")
    
    elif args.command == "ui":
        # Import here to avoid loading Gradio unnecessarily
        from app import CustomToneApp
        
        # Launch UI
        app = CustomToneApp(args.config)
        app.launch(share=args.share)
    
    else:
        logger.error("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main()
