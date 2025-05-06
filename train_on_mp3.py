# Example script for training the custom music generation model on MP3 files

# Install required packages
!pip install torch torchaudio librosa matplotlib soundfile glob2

# Import the model and required libraries
from custom_model import MusicGenerationConfig, MusicGenerator, MusicSourceFile
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to find MP3 files
def find_mp3_files(directory, limit=None):
    """Find all MP3 files in a directory and its subdirectories"""
    mp3_files = glob.glob(os.path.join(directory, "**/*.mp3"), recursive=True)
    if limit:
        mp3_files = mp3_files[:limit]
    return mp3_files

# Create a directory for training data
!mkdir -p training_data

# For demonstration, you would upload your MP3 files to the training_data directory
# In Google Colab, you can use:
# from google.colab import files
# uploaded = files.upload()  # This will prompt you to upload files
# 
# # Save uploaded files to the training_data directory
# for filename, content in uploaded.items():
#     with open(os.path.join('training_data', filename), 'wb') as f:
#         f.write(content)

# For this example, we'll assume MP3 files are already in the training_data directory
# Find all MP3 files
print("Finding MP3 files...")
mp3_files = find_mp3_files("training_data")
print(f"Found {len(mp3_files)} MP3 files")

if len(mp3_files) == 0:
    print("No MP3 files found. Please upload some MP3 files to the training_data directory.")
    # For demonstration, we'll create a dummy file list
    mp3_files = ["dummy_file_1.mp3", "dummy_file_2.mp3"]
    print("Using dummy file list for demonstration purposes.")

# Create a custom configuration
print("Creating configuration...")
config = MusicGenerationConfig()

# Customize configuration for training
config.batch_size = 8  # Smaller batch size to avoid memory issues
config.learning_rate = 1e-4  # Lower learning rate for stability
config.num_epochs = 50  # Fewer epochs for demonstration

# Initialize the generator
print("Initializing generator...")
generator = MusicGenerator(config)

# Train the model
print("Starting training...")
start_time = time.time()

try:
    generator.train(
        mp3_files,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size
    )
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Training failed: {e}")
    print("Continuing with demonstration using untrained model...")

# Save the trained model
print("Saving trained model...")
model_path = "trained_music_model.pt"
generator.save_model(model_path)
print(f"Model saved to {model_path}")

# Generate samples from the trained model
print("Generating samples from trained model...")
for i in range(3):
    print(f"Generating sample {i+1}...")
    
    # Create a source file with random instruments
    source_file = generator.create_source_file(random_seed=i)
    
    # Render audio
    combined, stems = generator.render_audio(source_file)
    
    # Play the combined audio
    print(f"Sample {i+1}:")
    display(Audio(combined, rate=config.sample_rate))
    
    # Save the audio
    output_path = f"generated_sample_{i+1}.wav"
    import soundfile as sf
    sf.write(output_path, combined, config.sample_rate)
    print(f"Sample saved to {output_path}")

# Demonstrate source file editing
print("\nDemonstrating source file editing...")

# Create a new source file
source_file = generator.create_source_file(random_seed=42)

# Generate the original version
print("Original version:")
combined_original, _ = generator.render_audio(source_file)
display(Audio(combined_original, rate=config.sample_rate))

# Modify the drums
print("Version with modified drums:")
source_file = generator.modify_instrument(source_file, "drums", "randomize", 0.5)
combined_drums, _ = generator.render_audio(source_file)
display(Audio(combined_drums, rate=config.sample_rate))

# Remove the vocals
print("Version without vocals:")
no_vocals = source_file.metadata["instrument_volumes"].copy()
no_vocals["vocals"] = 0.0
combined_no_vocals, _ = generator.render_audio(source_file, custom_volumes=no_vocals)
display(Audio(combined_no_vocals, rate=config.sample_rate))

# Create a version with only piano and strings
print("Version with only piano and strings:")
piano_strings = {instr: 0.0 for instr in source_file.metadata["instrument_volumes"]}
piano_strings["piano"] = 1.2
piano_strings["strings"] = 1.0
combined_piano_strings, _ = generator.render_audio(source_file, custom_volumes=piano_strings)
display(Audio(combined_piano_strings, rate=config.sample_rate))

# Save the source file
source_file_path = "editable_song.msf"
source_file.save(source_file_path)
print(f"Source file saved to {source_file_path}")

print("All examples completed successfully!")

# Advanced: Load the source file and make further modifications
print("\nAdvanced: Loading source file and making further modifications...")
loaded_source = MusicSourceFile.load(source_file_path, config)

# Modify multiple instruments
loaded_source = generator.modify_instrument(loaded_source, "synth", "interpolate", 0.7)
loaded_source = generator.modify_instrument(loaded_source, "guitar", "amplify", 0.4)

# Adjust volumes for a different mix
loaded_source.set_instrument_volume("drums", 0.6)
loaded_source.set_instrument_volume("bass", 1.1)
loaded_source.set_instrument_volume("synth", 1.3)

# Render the modified version
print("Modified version after loading:")
combined_modified, _ = generator.render_audio(loaded_source)
display(Audio(combined_modified, rate=config.sample_rate))

# Save the modified version
sf.write("modified_song.wav", combined_modified, config.sample_rate)
print("Modified version saved to modified_song.wav")

print("All advanced examples completed successfully!")