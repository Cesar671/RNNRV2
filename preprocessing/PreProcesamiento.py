import librosa
import librosa.display
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import medfilt
def preprocesar_audio(audio_path, threshold=30, hop_length=512):
    audio, sr = librosa.load(audio_path)

    audio = librosa.to_mono(audio)

    audio_normalizado = librosa.util.normalize(audio)

    audio_resampled = librosa.resample(audio_normalizado, orig_sr=librosa.get_samplerate(audio_path), target_sr=16000)

    audio_filtrado = medfilt(audio_resampled, kernel_size=3)

    audio_filtrado = np.int16(audio_filtrado * 32767)
    tramos_activos = librosa.effects.split(audio_filtrado, top_db=threshold, hop_length=hop_length)

    audio_sin_espacios, _ = librosa.effects.trim(audio_filtrado)

    """mfccs = []
    for inicio, fin in tramos_activos:
        audio_sin_espacios = []
        audio_sin_espacios.extend(audio_filtrado[inicio:fin])
        audio_sin_espacios = np.array(audio_sin_espacios)
        sf.write(audio_path, audio_sin_espacios, sr)
        mfccs.append(extraer_mfcc(audio_path))"""
    audio_sin_espacios = []
    for inicio, fin in tramos_activos:
        audio_sin_espacios.extend(audio_filtrado[inicio:fin])
    audio_sin_espacios = np.array(audio_sin_espacios)
    sf.write(audio_path, audio_sin_espacios, sr)
    return [extraer_mfcc(audio_path)]


def extraer_mfcc(file_path, window_size=0.1, step_size=0.05, mfcc_dim=20):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Calcular el número total de ventanas y el número de pasos entre ventanas
    window_length = int(window_size * sample_rate)
    step_length = int(step_size * sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=mfcc_dim, n_fft=window_length, hop_length=step_length).T
    return mfcc



def grabar_audio(duracion, sr=16000, nombre="grabacion_sample.wav"):
    print("#####...Grabando...####")
    audio = sd.rec(int(duracion * sr), samplerate=sr, channels=1)
    sd.wait()  # Espera a que la grabación termine
    print("Grabación completada.")
    sf.write(nombre, audio, sr)
    sf.write("grabado/raw.wav", audio, sr)


def rellenar_caracteristicas(datos, tam_max, mfcc_dim):
    tam = len(datos)
    padded_features = np.zeros((tam, tam_max, mfcc_dim))

    for i, feature in enumerate(datos):
        padded_features[i, :feature.shape[0], :] = feature

    return padded_features