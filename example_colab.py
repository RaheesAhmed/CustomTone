# Example script for using the custom music generation model in Google Colab

# Install required packages
!pip install torch torchaudio librosa matplotlib soundfile

# Import the model
from custom_model import MusicGenerationConfig, MusicGenerator, MusicSourceFile
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import numpy as np
import torch

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a new music generator
print("Initializing music generator...")
config = MusicGenerationConfig()
generator = MusicGenerator(config)

# Create a new source file with random instruments
print("Creating source file with random instruments...")
source_file = generator.create_source_file(random_seed=42)

# Display the available instruments
print(f"Available instruments: {config.instrument_tracks}")

# Modify instruments
print("Modifying instruments...")
source_file = generator.modify_instrument(source_file, "drums", "randomize", 0.2)
source_file = generator.modify_instrument(source_file, "bass", "amplify", 0.3)
source_file = generator.modify_instrument(source_file, "piano", "interpolate", 0.4)
source_file = generator.modify_instrument(source_file, "guitar", "attenuate", 0.2)

# Adjust volumes
print("Adjusting volumes...")
source_file.set_instrument_volume("drums", 0.8)
source_file.set_instrument_volume("bass", 1.2)
source_file.set_instrument_volume("vocals", 0.9)
source_file.set_instrument_volume("synth", 0.7)

# Render audio
print("Rendering audio...")
combined, stems = generator.render_audio(source_file)

# Play the combined audio
print("Playing combined audio:")
display(Audio(combined, rate=config.sample_rate))

# Play individual stems
print("Playing individual stems:")
for instrument, audio in stems.items():
    print(f"Playing {instrument}:")
    display(Audio(audio, rate=config.sample_rate))

# Visualize the audio waveforms
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

# Save the source file for later editing
print("Saving source file...")
source_file.save("my_song.msf")

# Save the audio
print("Saving audio file...")
import soundfile as sf
sf.write("my_song.wav", combined, config.sample_rate)

print("Example completed successfully!")

# Advanced: Create a version without vocals
print("\nCreating a version without vocals...")
no_vocals = source_file.metadata["instrument_volumes"].copy()
no_vocals["vocals"] = 0.0  # Mute vocals

combined_no_vocals, _ = generator.render_audio(source_file, custom_volumes=no_vocals)
print("Playing version without vocals:")
display(Audio(combined_no_vocals, rate=config.sample_rate))

# Save the instrumental version
sf.write("my_song_instrumental.wav", combined_no_vocals, config.sample_rate)

# Advanced: Create a version with only rhythm section
print("\nCreating a version with only rhythm section...")
rhythm_only = {instr: 0.0 for instr in source_file.metadata["instrument_volumes"]}
rhythm_only["drums"] = 1.0
rhythm_only["bass"] = 1.0

combined_rhythm, _ = generator.render_audio(source_file, custom_volumes=rhythm_only)
print("Playing rhythm section only:")
display(Audio(combined_rhythm, rate=config.sample_rate))

# Save the rhythm section version
sf.write("my_song_rhythm.wav", combined_rhythm, config.sample_rate)

print("All examples completed successfully!")