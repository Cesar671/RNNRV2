from reconocimiento.RNN import Rnn
from preprocessing.PreProcesamiento import grabar_audio, preprocesar_audio
import conversor.convertidor

def reconocer_voz():
    ruta_datos = "caracteristicas_audios.npz"
    ruta_audio = "grabado/Record1.wav"
    rnn = Rnn.RnnRecognizer(ruta_datos)

    # grabar la voz
    grabar_audio(5,nombre=ruta_audio)
    mfccs = preprocesar_audio(ruta_audio)

    # etapa de reconocimiento
    palabras = []
    for mfcc in mfccs:
        palabra = rnn.reconocer_palabra(mfcc)
        palabras.append(palabra)
    return palabras
