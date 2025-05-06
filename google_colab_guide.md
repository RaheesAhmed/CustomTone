# Using the Custom Music Generation Model in Google Colab

This guide provides step-by-step instructions for using the custom music generation model in Google Colab.

## Setup

1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Create a new notebook
3. Upload the following files to your Colab session:
   - `custom_model.py` (the main model)
   - `example_colab.py` (optional, for quick examples)
   - `train_on_mp3.py` (optional, for training on your own music)

To upload files, click on the folder icon in the left sidebar, then click the upload button.

## Quick Start: Generate Music

Run the following code to quickly generate music with the model:

```python
# Install required packages
!pip install torch torchaudio librosa matplotlib soundfile

# Import the model
from custom_model import MusicGenerationConfig, MusicGenerator
from IPython.display import Audio

# Create a new music generator
config = MusicGenerationConfig()
generator = MusicGenerator(config)

# Create a source file with random instruments
source_file = generator.create_source_file(random_seed=42)

# Render audio
combined, stems = generator.render_audio(source_file)

# Play the combined audio
Audio(combined, rate=config.sample_rate)
```

## Training on Your Own Music

To train the model on your own music collection:

1. Upload your MP3 files to Google Colab
2. Run the following code:

```python
# Install required packages
!pip install torch torchaudio librosa matplotlib soundfile glob2

# Import the model
from custom_model import MusicGenerationConfig, MusicGenerator
import glob
import os
from google.colab import files

# Create a directory for your music files
!mkdir -p my_music

# Upload MP3 files
uploaded = files.upload()

# Save uploaded files to the my_music directory
for filename, content in uploaded.items():
    with open(os.path.join('my_music', filename), 'wb') as f:
        f.write(content)

# Find all MP3 files
mp3_files = glob.glob('my_music/*.mp3')
print(f"Found {len(mp3_files)} MP3 files")

# Create a new music generator
config = MusicGenerationConfig()
# Customize configuration for training
config.batch_size = 8  # Smaller batch size to avoid memory issues
config.learning_rate = 1e-4  # Lower learning rate for stability
config.num_epochs = 50  # Adjust based on your needs

generator = MusicGenerator(config)

# Train the model
generator.train(
    mp3_files,
    num_epochs=config.num_epochs,
    learning_rate=config.learning_rate,
    batch_size=config.batch_size
)

# Save the trained model
generator.save_model("my_trained_model.pt")

# Generate a sample from the trained model
source_file = generator.create_source_file()
combined, _ = generator.render_audio(source_file)
Audio(combined, rate=config.sample_rate)
```

## Customizing Instruments

You can customize individual instruments in your generated music:

```python
# Create a source file
source_file = generator.create_source_file()

# Modify instruments
source_file = generator.modify_instrument(source_file, "drums", "randomize", 0.2)
source_file = generator.modify_instrument(source_file, "bass", "amplify", 0.3)
source_file = generator.modify_instrument(source_file, "piano", "interpolate", 0.4)
source_file = generator.modify_instrument(source_file, "guitar", "attenuate", 0.2)

# Adjust volumes
source_file.set_instrument_volume("drums", 0.8)
source_file.set_instrument_volume("bass", 1.2)
source_file.set_instrument_volume("vocals", 0.0)  # Mute vocals

# Render audio
combined, stems = generator.render_audio(source_file)
Audio(combined, rate=config.sample_rate)
```

## Creating Different Versions of a Song

You can create different versions of a song by adjusting which instruments are included and their volumes:

```python
# Create a source file
source_file = generator.create_source_file()

# Generate the full version
combined_full, stems = generator.render_audio(source_file)

# Create an instrumental version (no vocals)
no_vocals = source_file.metadata["instrument_volumes"].copy()
no_vocals["vocals"] = 0.0
combined_instrumental, _ = generator.render_audio(source_file, custom_volumes=no_vocals)

# Create a rhythm section only version
rhythm_only = {instr: 0.0 for instr in source_file.metadata["instrument_volumes"]}
rhythm_only["drums"] = 1.0
rhythm_only["bass"] = 1.0
combined_rhythm, _ = generator.render_audio(source_file, custom_volumes=rhythm_only)

# Play the different versions
print("Full version:")
Audio(combined_full, rate=config.sample_rate)

print("Instrumental version:")
Audio(combined_instrumental, rate=config.sample_rate)

print("Rhythm section only:")
Audio(combined_rhythm, rate=config.sample_rate)
```

## Saving and Loading Source Files

The source file format allows you to save your music and edit it later:

```python
# Save a source file
source_file.save("my_song.msf")

# Load a source file
from custom_model import MusicSourceFile
loaded_source = MusicSourceFile.load("my_song.msf", config)

# Make changes to the loaded source file
loaded_source = generator.modify_instrument(loaded_source, "synth", "interpolate", 0.7)
loaded_source.set_instrument_volume("drums", 0.6)

# Render the modified version
combined_modified, _ = generator.render_audio(loaded_source)
Audio(combined_modified, rate=config.sample_rate)
```

## Exporting Audio Files

You can export your generated music as WAV files:

```python
import soundfile as sf

# Render audio
combined, stems = generator.render_audio(source_file)

# Save the combined audio
sf.write("my_song.wav", combined, config.sample_rate)

# Save individual stems
for instrument, audio in stems.items():
    sf.write(f"my_song_{instrument}.wav", audio, config.sample_rate)
```

## Advanced: Visualizing the Audio

You can visualize the audio waveforms and spectrograms:

```python
import matplotlib.pyplot as plt
import numpy as np

# Render audio
combined, stems = generator.render_audio(source_file)

# Plot waveforms
plt.figure(figsize=(12, 8))
plt.subplot(len(stems) + 1, 1, 1)
plt.plot(combined)
plt.title("Combined Audio")
plt.ylim(-1, 1)

for i, (instrument, audio) in enumerate(stems.items(), 2):
    plt.subplot(len(stems) + 1, 1, i)
    plt.plot(audio)
    plt.title(f"{instrument} Audio")
    plt.ylim(-1, 1)

plt.tight_layout()
plt.show()

# Plot spectrograms
from librosa import display as libdisplay

plt.figure(figsize=(12, 8))
for i, (instrument, audio) in enumerate(stems.items(), 1):
    plt.subplot(len(stems), 1, i)
    mel_spec = generator.processor.extract_mel_spectrogram(audio)
    libdisplay.specshow(
        mel_spec,
        sr=config.sample_rate,
        hop_length=config.hop_length,
        x_axis='time',
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{instrument} Spectrogram")

plt.tight_layout()
plt.show()
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors, try:

1. Reducing the batch size: `config.batch_size = 4`
2. Reducing the model size: `config.hidden_dim = 256`, `config.latent_dim = 128`
3. Using a smaller segment length for training: `dataset = MusicDataset(audio_files, config, segment_length=3)`

### Slow Training

If training is too slow:

1. Reduce the number of epochs: `config.num_epochs = 20`
2. Use a smaller dataset for initial experiments
3. Use Google Colab with GPU acceleration (Runtime > Change runtime type > GPU)

### Poor Audio Quality

If the generated audio has poor quality:

1. Train on more data
2. Train for more epochs
3. Experiment with different modification parameters
4. Try different random seeds: `source_file = generator.create_source_file(random_seed=123)`

## Next Steps

Once you're comfortable with the basic usage, you can:

1. Experiment with different instrument combinations
2. Train on specific genres of music
3. Modify the model architecture in `custom_model.py` to suit your needs
4. Integrate with other audio processing libraries for more advanced effects
