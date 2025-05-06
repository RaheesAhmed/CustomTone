import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import json
import time
from tqdm.notebook import tqdm
import random
import pickle
import warnings
warnings.filterwarnings('ignore')

class MusicGenerationConfig:
    """Configuration for the music generation model"""
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        # Model parameters
        self.hidden_dim = 512
        self.latent_dim = 256
        self.num_layers = 4
        self.dropout = 0.1
        
        # Training parameters
        self.batch_size = 16
        self.learning_rate = 3e-4
        self.num_epochs = 100
        
        # Generation parameters
        self.max_sequence_length = 1024
        self.temperature = 1.0
        
        # Instrument tracks (can be customized)
        self.instrument_tracks = [
            "drums", "bass", "piano", "guitar", "strings", "synth", "vocals"
        ]
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def save(self, path):
        """Save configuration to a JSON file"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load(cls, path):
        """Load configuration from a JSON file"""
        config = cls()
        with open(path, 'r') as f:
            config.__dict__.update(json.load(f))
        return config

class AudioProcessor:
    """Process audio files for training and generation"""
    def __init__(self, config):
        self.config = config
        
    def load_audio(self, file_path):
        """Load audio file and convert to mono if needed"""
        try:
            waveform, sr = torchaudio.load(file_path)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)
                
            return waveform.squeeze(0).numpy()
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
    
    def extract_mel_spectrogram(self, waveform):
        """Convert waveform to mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, 
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec
    
    def normalize_spectrogram(self, spectrogram):
        """Normalize spectrogram to range [0, 1]"""
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        return (spectrogram - min_val) / (max_val - min_val)
    
    def spectrogram_to_audio(self, spectrogram):
        """Convert spectrogram back to audio waveform"""
        # Denormalize if needed
        if spectrogram.max() <= 1.0 and spectrogram.min() >= 0.0:
            spectrogram = spectrogram * 80 - 80  # Approximate dB range
        
        # Convert from dB to power
        mel_spec = librosa.db_to_power(spectrogram)
        
        # Mel spectrogram to audio
        waveform = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        return waveform
    
    def separate_stems(self, waveform):
        """
        Separate audio into different instrument stems
        This is a simplified version - in production, you would use
        a more sophisticated source separation model like Spleeter or Demucs
        """
        # For demonstration, we'll create dummy stems
        # In a real implementation, you would use a proper source separation model
        stems = {}
        for instrument in self.config.instrument_tracks:
            # This is just a placeholder - real separation would go here
            stems[instrument] = waveform * np.random.uniform(0.5, 1.0)
            
        return stems
    
    def combine_stems(self, stems, volumes=None):
        """Combine stems into a single audio track with optional volume control"""
        if volumes is None:
            volumes = {instr: 1.0 for instr in stems.keys()}
            
        combined = np.zeros_like(list(stems.values())[0])
        for instr, audio in stems.items():
            if instr in volumes:
                combined += audio * volumes[instr]
                
        # Normalize to prevent clipping
        max_val = np.max(np.abs(combined))
        if max_val > 1.0:
            combined = combined / max_val
            
        return combined

class MusicDataset(Dataset):
    """Dataset for training the music generation model"""
    def __init__(self, audio_files, config, transform=None, segment_length=5):
        self.audio_files = audio_files
        self.config = config
        self.transform = transform
        self.processor = AudioProcessor(config)
        self.segment_length = segment_length  # in seconds
        self.segment_samples = segment_length * config.sample_rate
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self.processor.load_audio(audio_path)
        
        if waveform is None or len(waveform) < self.segment_samples:
            # If audio is too short, pad with zeros
            if waveform is None:
                waveform = np.zeros(self.segment_samples)
            else:
                padding = np.zeros(self.segment_samples - len(waveform))
                waveform = np.concatenate([waveform, padding])
        else:
            # Randomly select a segment
            start = random.randint(0, len(waveform) - self.segment_samples)
            waveform = waveform[start:start + self.segment_samples]
        
        # Extract mel spectrogram
        mel_spec = self.processor.extract_mel_spectrogram(waveform)
        mel_spec = self.processor.normalize_spectrogram(mel_spec)
        
        # Convert to tensor
        mel_spec = torch.from_numpy(mel_spec).float()
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
            
        return mel_spec
class EncoderBlock(nn.Module):
    """Transformer encoder block for music generation"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class MusicEncoder(nn.Module):
    """Encoder for converting mel spectrograms to latent representations"""
    def __init__(self, config):
        super(MusicEncoder, self).__init__()
        self.config = config
        
        # Convolutional layers for downsampling
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Calculate the size after convolutions
        self.conv_output_size = self._get_conv_output_size()
        
        # Projection to hidden dimension
        self.projection = nn.Linear(self.conv_output_size, config.hidden_dim)
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(config.hidden_dim, num_heads=8, dropout=config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Projection to latent space
        self.to_latent = nn.Linear(config.hidden_dim, config.latent_dim)
        
    def _get_conv_output_size(self):
        # Helper function to calculate the size after convolutions
        with torch.no_grad():
            # Create a dummy input
            x = torch.zeros(1, 1, self.config.n_mels, self.config.max_sequence_length // self.config.hop_length)
            x = self.conv_layers(x)
            return x.numel() // x.shape[0]  # Total elements divided by batch size
        
    def forward(self, x):
        # Input shape: [batch_size, n_mels, time]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, n_mels, time]
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten and project
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.projection(x)
        
        # Reshape for transformer: [sequence_length, batch_size, hidden_dim]
        # For simplicity, we'll use a fixed sequence length of 1
        x = x.unsqueeze(0)
        
        # Apply transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
            
        # Project to latent space
        x = self.to_latent(x)
        
        # Output shape: [1, batch_size, latent_dim]
        return x.squeeze(0)  # [batch_size, latent_dim]

class MusicDecoder(nn.Module):
    """Decoder for generating mel spectrograms from latent representations"""
    def __init__(self, config):
        super(MusicDecoder, self).__init__()
        self.config = config
        
        # Projection from latent to hidden dimension
        self.from_latent = nn.Linear(config.latent_dim, config.hidden_dim)
        
        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            EncoderBlock(config.hidden_dim, num_heads=8, dropout=config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Calculate the size before deconvolutions
        self.conv_input_size = self._get_conv_input_size()
        
        # Projection to convolutional input
        self.to_conv = nn.Linear(config.hidden_dim, self.conv_input_size)
        
        # Deconvolutional layers for upsampling
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )
        
    def _get_conv_input_size(self):
        # Helper function to calculate the size before deconvolutions
        # This should match the output size of the encoder's convolutional layers
        return 256 * (self.config.n_mels // 16) * (self.config.max_sequence_length // (self.config.hop_length * 16))
        
    def forward(self, z, instrument_conditioning=None):
        # Input shape: [batch_size, latent_dim]
        
        # Project from latent to hidden dimension
        x = self.from_latent(z)
        
        # Add instrument conditioning if provided
        if instrument_conditioning is not None:
            x = x + instrument_conditioning
        
        # Reshape for transformer: [sequence_length, batch_size, hidden_dim]
        x = x.unsqueeze(0)
        
        # Apply transformer decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
            
        # Project to convolutional input
        x = self.to_conv(x.squeeze(0))
        
        # Reshape for deconvolution
        batch_size = x.shape[0]
        x = x.view(batch_size, 256, self.config.n_mels // 16, self.config.max_sequence_length // (self.config.hop_length * 16))
        
        # Apply deconvolutional layers
        x = self.deconv_layers(x)
        
        # Output shape: [batch_size, 1, n_mels, time]
        return x.squeeze(1)  # [batch_size, n_mels, time]

class InstrumentEncoder(nn.Module):
    """Encoder for instrument-specific conditioning"""
    def __init__(self, config):
        super(InstrumentEncoder, self).__init__()
        self.config = config
        num_instruments = len(config.instrument_tracks)
        
        # Embedding for instruments
        self.instrument_embedding = nn.Embedding(num_instruments, config.hidden_dim)
        
    def forward(self, instrument_idx):
        # Input: instrument index
        # Output: instrument embedding [batch_size, hidden_dim]
        return self.instrument_embedding(instrument_idx)

class MusicVAE(nn.Module):
    """Variational Autoencoder for music generation with instrument separation"""
    def __init__(self, config):
        super(MusicVAE, self).__init__()
        self.config = config
        
        # Encoder and decoder
        self.encoder = MusicEncoder(config)
        self.decoder = MusicDecoder(config)
        
        # Instrument encoder
        self.instrument_encoder = InstrumentEncoder(config)
        
        # VAE components
        self.fc_mu = nn.Linear(config.latent_dim, config.latent_dim)
        self.fc_var = nn.Linear(config.latent_dim, config.latent_dim)
        
        # Instrument-specific latent projections
        self.instrument_projections = nn.ModuleDict({
            instr: nn.Linear(config.latent_dim, config.latent_dim)
            for instr in config.instrument_tracks
        })
        
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, instrument=None):
        """Decode latent vector to output"""
        # Apply instrument-specific projection if specified
        if instrument is not None:
            instrument_idx = self.config.instrument_tracks.index(instrument)
            instrument_idx = torch.tensor([instrument_idx], device=z.device)
            instrument_cond = self.instrument_encoder(instrument_idx)
            z = self.instrument_projections[instrument](z)
            return self.decoder(z, instrument_cond)
        else:
            return self.decoder(z)
    
    def forward(self, x, instrument=None):
        """Forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, instrument)
        return x_recon, mu, log_var
    
    def generate(self, num_samples=1, instrument=None, z=None, temperature=1.0):
        """Generate new samples"""
        if z is None:
            # Sample from prior
            z = torch.randn(num_samples, self.config.latent_dim, device=self.config.device) * temperature
        
        with torch.no_grad():
            samples = self.decode(z, instrument)
        return samples
class MusicSourceFile:
    """Source file format for storing and editing music generation"""
    def __init__(self, config):
        self.config = config
        self.latent_vectors = {}  # Instrument -> latent vector
        self.metadata = {
            "title": "Untitled",
            "created_at": time.time(),
            "modified_at": time.time(),
            "version": "1.0",
            "instrument_volumes": {instr: 1.0 for instr in config.instrument_tracks}
        }
        
    def add_instrument(self, instrument, latent_vector):
        """Add or update an instrument track"""
        if instrument not in self.config.instrument_tracks:
            raise ValueError(f"Unknown instrument: {instrument}")
        
        self.latent_vectors[instrument] = latent_vector.detach().cpu()
        self.metadata["modified_at"] = time.time()
        
    def remove_instrument(self, instrument):
        """Remove an instrument track"""
        if instrument in self.latent_vectors:
            del self.latent_vectors[instrument]
            self.metadata["modified_at"] = time.time()
            
    def set_instrument_volume(self, instrument, volume):
        """Set the volume for an instrument"""
        if instrument not in self.config.instrument_tracks:
            raise ValueError(f"Unknown instrument: {instrument}")
        
        self.metadata["instrument_volumes"][instrument] = max(0.0, min(1.0, volume))
        self.metadata["modified_at"] = time.time()
        
    def save(self, path):
        """Save source file to disk"""
        data = {
            "latent_vectors": {k: v.numpy() for k, v in self.latent_vectors.items()},
            "metadata": self.metadata,
            "config": {
                "instrument_tracks": self.config.instrument_tracks,
                "latent_dim": self.config.latent_dim
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    @classmethod
    def load(cls, path, config):
        """Load source file from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        source_file = cls(config)
        source_file.metadata = data["metadata"]
        
        # Convert numpy arrays back to tensors
        source_file.latent_vectors = {
            k: torch.from_numpy(v) for k, v in data["latent_vectors"].items()
        }
        
        return source_file

class MusicGenerator:
    """Main class for music generation and manipulation"""
    def __init__(self, config=None):
        if config is None:
            self.config = MusicGenerationConfig()
        else:
            self.config = config
            
        self.model = MusicVAE(self.config).to(self.config.device)
        self.processor = AudioProcessor(self.config)
        self.optimizer = None
        
    def train(self, audio_files, num_epochs=None, learning_rate=None, batch_size=None):
        """Train the model on a collection of audio files"""
        if num_epochs is not None:
            self.config.num_epochs = num_epochs
        if learning_rate is not None:
            self.config.learning_rate = learning_rate
        if batch_size is not None:
            self.config.batch_size = batch_size
            
        # Create dataset and dataloader
        dataset = MusicDataset(audio_files, self.config)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            recon_loss = 0.0
            kl_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in progress_bar:
                # Move batch to device
                batch = batch.to(self.config.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(batch)
                
                # Reconstruction loss (MSE)
                r_loss = F.mse_loss(recon_batch, batch)
                
                # KL divergence
                k_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                k_loss = k_loss / batch.size(0)  # Normalize by batch size
                
                # Total loss
                loss = r_loss + k_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                recon_loss += r_loss.item()
                kl_loss += k_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'recon': r_loss.item(),
                    'kl': k_loss.item()
                })
                
            # Print epoch summary
            avg_loss = epoch_loss / len(dataloader)
            avg_recon = recon_loss / len(dataloader)
            avg_kl = kl_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
            
            # Generate a sample every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.generate_sample(epoch + 1)
                
    def generate_sample(self, epoch=None):
        """Generate and display a sample during training"""
        self.model.eval()
        with torch.no_grad():
            # Generate a random sample
            z = torch.randn(1, self.config.latent_dim, device=self.config.device)
            sample = self.model.generate(z=z)[0].cpu().numpy()
            
            # Convert to audio
            audio = self.processor.spectrogram_to_audio(sample)
            
            # Display spectrogram
            plt.figure(figsize=(10, 4))
            plt.imshow(sample, aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Generated Sample (Epoch {epoch})" if epoch else "Generated Sample")
            plt.tight_layout()
            plt.show()
            
            # Play audio
            display(Audio(audio, rate=self.config.sample_rate))
            
        self.model.train()
        
    def save_model(self, path):
        """Save model weights to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config.__dict__
        }, path)
        
    def load_model(self, path):
        """Load model weights from disk"""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        # Update config if needed
        self.config.__dict__.update(checkpoint['config'])
        
        # Reinitialize model with updated config
        self.model = MusicVAE(self.config).to(self.config.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    def create_source_file(self, audio_path=None, random_seed=None):
        """Create a new source file from audio or random seed"""
        source_file = MusicSourceFile(self.config)
        
        if audio_path:
            # Load and process audio
            waveform = self.processor.load_audio(audio_path)
            
            # Separate stems
            stems = self.processor.separate_stems(waveform)
            
            # Encode each stem
            self.model.eval()
            with torch.no_grad():
                for instrument, audio in stems.items():
                    # Convert to mel spectrogram
                    mel_spec = self.processor.extract_mel_spectrogram(audio)
                    mel_spec = self.processor.normalize_spectrogram(mel_spec)
                    mel_spec = torch.from_numpy(mel_spec).float().to(self.config.device)
                    
                    # Encode to latent space
                    mu, _ = self.model.encode(mel_spec.unsqueeze(0))
                    source_file.add_instrument(instrument, mu[0])
        else:
            # Create random latent vectors
            if random_seed is not None:
                torch.manual_seed(random_seed)
                
            for instrument in self.config.instrument_tracks:
                z = torch.randn(self.config.latent_dim, device=self.config.device)
                source_file.add_instrument(instrument, z)
                
        return source_file
    
    def render_audio(self, source_file, output_path=None, include_instruments=None, custom_volumes=None):
        """Render audio from a source file"""
        if include_instruments is None:
            include_instruments = list(source_file.latent_vectors.keys())
            
        if custom_volumes is None:
            custom_volumes = source_file.metadata["instrument_volumes"]
            
        # Generate audio for each instrument
        self.model.eval()
        stems = {}
        with torch.no_grad():
            for instrument in include_instruments:
                if instrument in source_file.latent_vectors:
                    # Get latent vector
                    z = source_file.latent_vectors[instrument].to(self.config.device).unsqueeze(0)
                    
                    # Generate spectrogram
                    spec = self.model.decode(z, instrument)[0].cpu().numpy()
                    
                    # Convert to audio
                    audio = self.processor.spectrogram_to_audio(spec)
                    stems[instrument] = audio
        
        # Combine stems with volume control
        volumes = {instr: custom_volumes.get(instr, 1.0) for instr in stems.keys()}
        combined = self.processor.combine_stems(stems, volumes)
        
        # Save to file if path provided
        if output_path:
            import soundfile as sf
            sf.write(output_path, combined, self.config.sample_rate)
            
        return combined, stems
    
    def modify_instrument(self, source_file, instrument, modification_type, amount=0.1):
        """Modify an instrument track in the source file"""
        if instrument not in source_file.latent_vectors:
            raise ValueError(f"Instrument {instrument} not found in source file")
            
        z = source_file.latent_vectors[instrument].clone()
        
        if modification_type == "randomize":
            # Add random noise to the latent vector
            noise = torch.randn_like(z) * amount
            z = z + noise
        elif modification_type == "interpolate":
            # Interpolate towards a random point
            target = torch.randn_like(z)
            z = z * (1 - amount) + target * amount
        elif modification_type == "amplify":
            # Amplify the latent vector
            z = z * (1 + amount)
        elif modification_type == "attenuate":
            # Attenuate the latent vector
            z = z * (1 - amount)
        else:
            raise ValueError(f"Unknown modification type: {modification_type}")
            
        # Update the source file
        source_file.add_instrument(instrument, z)
        
        return source_file

# Example usage in Google Colab
def run_example():
    """Run an example of the music generation model in Google Colab"""
    # Create configuration
    config = MusicGenerationConfig()
    
    # Initialize generator
    generator = MusicGenerator(config)
    
    # Create a source file with random instruments
    source_file = generator.create_source_file(random_seed=42)
    
    # Modify some instruments
    source_file = generator.modify_instrument(source_file, "drums", "randomize", 0.2)
    source_file = generator.modify_instrument(source_file, "bass", "amplify", 0.3)
    
    # Adjust volumes
    source_file.set_instrument_volume("drums", 0.8)
    source_file.set_instrument_volume("bass", 1.2)
    
    # Render audio
    combined, stems = generator.render_audio(source_file)
    
    # Play the combined audio
    display(Audio(combined, rate=config.sample_rate))
    
    # Play individual stems
    for instrument, audio in stems.items():
        print(f"Playing {instrument}:")
        display(Audio(audio, rate=config.sample_rate))
        
    # Save the source file
    source_file.save("example_song.msf")
    
    # Save the audio
    import soundfile as sf
    sf.write("example_song.wav", combined, config.sample_rate)
    
    print("Example completed successfully!")

# Main execution
if __name__ == "__main__":
    # If running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
        
    if IN_COLAB:
        # Install required packages
        !pip install torch torchaudio librosa matplotlib soundfile
        
        # Run the example
        run_example()
    else:
        print("This script is designed to run in Google Colab.")
        print("You can still use the classes and functions directly.")