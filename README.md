# CustomTone: Customizable Music Generation System

![CustomTone Logo](https://img.shields.io/badge/CustomTone-Music%20Generation-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

CustomTone is an open-source music generation system inspired by SUNO but with enhanced customization capabilities. Unlike other music generation systems that produce a single audio file, CustomTone provides:

1. **Full Song Generation**: Create complete songs from text prompts or lyrics
2. **Stem Separation**: Access individual instrument tracks (vocals, drums, bass, other)
3. **MIDI Transcription**: Convert audio to MIDI for note-level editing
4. **Customizable Mixing**: Add, remove, or modify individual instruments
5. **Source File Access**: Edit your music like a designer edits in Adobe Illustrator

## Architecture

CustomTone uses a modular architecture with three main components:

1. **Generation**: YuE or AudioCraft/MusicGen for high-quality music generation
2. **Separation**: Demucs for splitting audio into instrument stems
3. **Transcription**: BasicPitch for converting audio to editable MIDI

![Architecture Diagram](https://mermaid.ink/img/pako:eNp1kU1PwzAMhv9KlBOgSf3YpE1w2A6cEBJiB8QOaRo2o6ZJlTgwTeq_4-4DwYYvsd-8fmwnB2WNRpWpnT0YfHK4Jvhw3pAJZOkFXOGWnIcXbMHBGjsKYIAtNOQpwJbcAB15ik_Yk6MtmR4-yVuADjRZCmvoKKyxI9-DxY7sBpZkLQxkLXXkLDgKa-zIW-gpbLAn38OGwhp7Cj1YCmscyFsYKKxxoNDDQGGNkcIGRwpr3FHYwI7CGiOFDY4UNjhS2MBI3sKewgYjhQ3uKWxwT2EDe_IWDhQ2eKCwgQOFDRzIWzhS2OCRwgYfKWzwicIGn8lbOFLY4DOFDb5Q2OALhQ2-UtjgK4UNvpG3cKKwwTcKG3ynsIEThQ1-kLdwprDBDwobPFPY4CeFDX6St_CLwgZ_KGzwl8IGfylscKawwV_yFv5R2OCZwgbPFDb4R2GD_8hbOFPY4D-FDf5T2OCZwgb_yVv4D5ZV0Qs?type=png)

## Features

- **Text-to-Music Generation**: Create songs from text descriptions or lyrics
- **Style Control**: Specify genre, mood, tempo, and instrumentation
- **Stem Access**: Get separate tracks for vocals, drums, bass, and other instruments
- **MIDI Export**: Edit notes, timing, and pitch in any DAW
- **Customizable Mixing**: Select which instruments to include in the final mix
- **Kaggle Integration**: Optimized to run on Kaggle's GPU environment
- **Gradio Interface**: User-friendly web interface for easy interaction

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 8+ GB VRAM for YuE model
- 4+ GB VRAM for MusicGen model

## Quick Start

1. Clone this repository:

```bash
git clone https://github.com/raheesahmed/CustomTone.git
cd CustomTone
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the Gradio interface:

```bash
python app.py
```

4. For Kaggle:
   - Upload the notebook `CustomTone_Kaggle.ipynb`
   - Enable GPU acceleration
   - Run all cells

## Usage

### Basic Usage

1. Enter lyrics or a text description
2. Select a musical style
3. Generate the full song
4. Access individual stems (vocals, drums, bass, other)
5. Download stems or MIDI files for editing
6. Recombine selected stems for a customized mix

### Advanced Usage

- **MIDI Editing**: Import MIDI files into your DAW for detailed editing
- **Custom Training**: Train the model on your own music collection (see docs/training.md)
- **API Integration**: Use the Python API for batch processing (see docs/api.md)

## Project Structure

```
CustomTone/
├── app.py                  # Gradio web interface
├── main.py                 # Command-line interface
├── config.yaml             # Configuration settings
├── requirements.txt        # Dependencies
├── CustomTone_Kaggle.ipynb # Kaggle notebook
├── model/
│   ├── music_generator.py  # Music generation module
│   ├── stem_separator.py   # Audio separation module
│   ├── midi_transcriber.py # Audio-to-MIDI conversion
│   └── utils.py            # Utility functions
├── data/
│   └── examples/           # Example prompts and outputs
└── docs/
    ├── api.md              # API documentation
    └── training.md         # Custom training guide
```

## Models

CustomTone supports multiple music generation models:

1. **YuE**: Open-source full-song generation model (similar to SUNO)
2. **MusicGen**: Meta's AudioCraft music generation model
3. **Magenta**: Google's music generation framework (optional)

For stem separation, we use:

- **Demucs**: Facebook's high-quality audio source separation
- **Spleeter**: Deezer's stem separator (alternative)

For MIDI transcription:

- **BasicPitch**: Spotify's audio-to-MIDI converter

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YuE](https://github.com/multimodal-art-projection/YuE) by Multimodal Art Projection team
- [AudioCraft](https://github.com/facebookresearch/audiocraft) by Meta AI
- [Demucs](https://github.com/facebookresearch/demucs) by Facebook Research
- [BasicPitch](https://github.com/spotify/basic-pitch) by Spotify
