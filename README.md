# Custom Music Generation Model

This project implements a custom music generation model that allows for separate instrument tracks and customization, similar to SUNO but with more control over individual elements.

## Features

- Generate music with separate instrument tracks (drums, bass, piano, guitar, strings, synth, vocals)
- Customize each instrument track independently
- Adjust volume levels for each instrument
- Save and load "source files" that can be edited later
- Train on your own music collection
- Run in Google Colab with minimal setup

## How It Works

The model uses a Variational Autoencoder (VAE) architecture with instrument-specific conditioning to generate music. The key components are:

1. **Audio Processing**: Convert audio files to mel spectrograms and back
2. **Instrument Separation**: Separate audio into different instrument stems
3. **Latent Space**: Encode music into a compact latent representation
4. **Source File Format**: Store latent vectors for each instrument track
5. **Customization**: Modify instrument tracks by manipulating latent vectors

## Getting Started

### Running in Google Colab

1. Upload the `custom_model.py` file to Google Colab
2. Run the following code:

```python
# Import the model
from custom_model import MusicGenerationConfig, MusicGenerator, MusicSourceFile

# Create a new music generator
config = MusicGenerationConfig()
generator = MusicGenerator(config)

# Create a new source file with random instruments
source_file = generator.create_source_file(random_seed=42)

# Modify instruments as desired
source_file = generator.modify_instrument(source_file, "drums", "randomize", 0.2)
source_file = generator.modify_instrument(source_file, "bass", "amplify", 0.3)

# Adjust volumes
source_file.set_instrument_volume("drums", 0.8)
source_file.set_instrument_volume("bass", 1.2)

# Render audio
combined, stems = generator.render_audio(source_file)

# Play the combined audio
from IPython.display import Audio
Audio(combined, rate=config.sample_rate)

# Save the source file for later editing
source_file.save("my_song.msf")

# Save the audio
import soundfile as sf
sf.write("my_song.wav", combined, config.sample_rate)
```

### Training on Your Own Music

To train the model on your own music collection:

```python
from custom_model import MusicGenerationConfig, MusicGenerator
import glob

# Get a list of audio files
audio_files = glob.glob("path/to/your/music/*.mp3")

# Create a new music generator
config = MusicGenerationConfig()
generator = MusicGenerator(config)

# Train the model
generator.train(
    audio_files,
    num_epochs=100,
    learning_rate=3e-4,
    batch_size=16
)

# Save the trained model
generator.save_model("my_trained_model.pt")
```

## Customization Options

### Instrument Modifications

- **Randomize**: Add random noise to the instrument's latent vector
- **Interpolate**: Blend the instrument with a random variation
- **Amplify**: Increase the intensity of the instrument
- **Attenuate**: Decrease the intensity of the instrument

### Volume Control

Adjust the volume of each instrument independently:

```python
source_file.set_instrument_volume("drums", 0.8)  # 80% volume
source_file.set_instrument_volume("bass", 1.2)   # 120% volume
source_file.set_instrument_volume("vocals", 0.0) # Mute vocals
```

### Adding/Removing Instruments

```python
# Remove an instrument
source_file.remove_instrument("vocals")

# Add a new instrument (from another source file)
other_source = generator.create_source_file()
source_file.add_instrument("vocals", other_source.latent_vectors["vocals"])
```

## Requirements

- Python 3.6+
- PyTorch
- Torchaudio
- Librosa
- Matplotlib
- Soundfile

## Limitations

- The current implementation uses a simplified instrument separation approach
- For production use, you would want to integrate a more sophisticated source separation model like Spleeter or Demucs
- Training requires a substantial amount of data for best results

## Future Improvements

- Integration with professional source separation models
- More sophisticated latent space manipulation
- User interface for visual editing
- Support for MIDI import/export
- Fine-grained control over musical elements (rhythm, harmony, melody)
