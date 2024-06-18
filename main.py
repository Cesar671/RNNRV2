from RV.RVRNN import reconocer_voz
#from grabado import grabador


def main():
    palabra = reconocer_voz()
    print(f"La palabra reconocida es {palabra}")

if __name__ == "__main__":
    main()
