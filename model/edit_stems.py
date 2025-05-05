import pretty_midi
from pydub import AudioSegment
import os

# Edit MIDI tracks
def remove_instrument_from_midi(midi_path, instrument_name, output_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    filtered_instruments = [inst for inst in midi_data.instruments if instrument_name.lower() not in pretty_midi.program_to_instrument_name(inst.program).lower()]
    midi_data.instruments = filtered_instruments
    midi_data.write(output_path)
    return output_path

# Combine selected stems
def combine_stems(selected_stems, output_path="generated/audio/final_mix.wav"):
    combined = AudioSegment.silent(duration=0)
    for stem in selected_stems:
        if os.path.exists(stem):
            combined += AudioSegment.from_wav(stem)
    combined.export(output_path, format="wav")
    return output_path