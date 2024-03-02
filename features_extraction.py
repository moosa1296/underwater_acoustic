import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def process_and_save_files(folder_path, output_dir, feature_type='spectrogram'):
    
    for species_name in os.listdir(folder_path):
        species_path = os.path.join(folder_path, species_name)
        if os.path.isdir(species_path):
            species_output_dir = os.path.join(output_dir, species_name)
            os.makedirs(species_output_dir, exist_ok=True)
            
            for audio_file in os.listdir(species_path):
                if audio_file.lower().endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(species_path, audio_file)
                    y, sr = librosa.load(audio_path)
                    if feature_type == 'spectrogram':
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                        S_dB = librosa.power_to_db(S, ref=np.max)
                        fig, ax = plt.subplots()
                        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
                        fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    elif feature_type == 'mfcc':
                        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        fig, ax = plt.subplots()
                        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
                        fig.colorbar(img, ax=ax)
                    else:
                        raise ValueError("Unsupported feature type.")
                    
                    output_path = os.path.join(species_output_dir, audio_file.rsplit('.', 1)[0] + '.png')
                    fig.savefig(output_path)
                    plt.close(fig)


folder_path = '/home/user-1/underwater_dataset'
output_dir = '/home/user-1/underwater_spectrograms'
process_and_save_files(folder_path, output_dir, feature_type='spectrogram') 