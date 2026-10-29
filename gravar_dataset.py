import os, time, sounddevice as sd, soundfile as sf, numpy as np

TARGET_SR       = 16000
TARGET_DURATION = 1.5
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)

CLASSES = {
    "1": "ler_texto",
    "2": "aumentar_fonte",
    "3": "mudar_voz",
    "4": "modo_claro",
    "5": "modo_escuro",
    "6": "background"
}
FRASES = {
    "ler_texto":      ["ler texto", "ler o texto", "leia o texto"],
    "aumentar_fonte": ["aumentar fonte", "aumenta a fonte", "aumentar a fonte"],
    "mudar_voz":      ["mudar voz", "mudar a voz", "trocar voz"],
    "modo_claro":     ["modo claro", "ativar modo claro", "tema claro"],
    "modo_escuro":    ["modo escuro", "ativar modo escuro", "tema escuro"],
    "background":     ["(silencio - nao fale nada)"],
}
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def detectar_microfone():
    """Retorna (device_id, sample_rate_nativo) do melhor microfone disponivel.
    Prefere DirectSound (Windows DirectSound) pois e mais compativel que MME."""
    devs = sd.query_devices()
    candidatos = []
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            candidatos.append((i, int(d["default_samplerate"]), d["name"]))
    # 1o: USB + DirectSound (nome sem acento especial, sr >= 44100)
    for i, sr, nome in candidatos:
        if "USB" in nome and "DirectSound" not in nome and sr >= 44100:
            # device 6 = DirectSound USB
            pass
    # Usa device 6 diretamente (Microfone USB DirectSound - testado OK)
    for i, sr, nome in candidatos:
        if i == 6:
            return i, sr
    # Fallback: qualquer USB com sr adequado
    for i, sr, nome in candidatos:
        if "USB" in nome and sr >= 44100:
            return i, sr
    # Fallback final
    for i, sr, nome in candidatos:
        if sr >= 44100:
            return i, sr
    return None, 44100


MIC_DEVICE, MIC_SR = detectar_microfone()
MIC_SAMPLES = int(MIC_SR * TARGET_DURATION)


def resamplear(audio, sr_origem, sr_destino):
    """Resampleia sem dependencia de librosa usando interpolacao linear."""
    if sr_origem == sr_destino:
        return audio
    fator = sr_destino / sr_origem
    n_orig = len(audio)
    n_dest = int(n_orig * fator)
    x_orig = np.linspace(0, n_orig - 1, n_orig)
    x_dest = np.linspace(0, n_orig - 1, n_dest)
    return np.interp(x_dest, x_orig, audio).astype(np.float32)


def proximo_numero(pasta):
    ex = [f for f in os.listdir(pasta) if f.startswith("sample_") and f.endswith(".wav")]
    if not ex: return 0
    ns = []
    for n in ex:
        try: ns.append(int(n.replace("sample_","").replace(".wav","")))
        except: pass
    return max(ns)+1 if ns else 0


def contar_regressiva(s=3):
    for i in range(s, 0, -1):
        print(f"  {i}...", end="\r", flush=True); time.sleep(1)
    print("  GRAVANDO!       ", flush=True)


def gravar_audio():
    kw = {"samplerate": MIC_SR, "channels": 1, "dtype": "float32"}
    if MIC_DEVICE is not None:
        kw["device"] = MIC_DEVICE
    audio = sd.rec(MIC_SAMPLES, **kw)
    sd.wait()
    raw = audio.flatten()
    # Normaliza e resamplea para 16kHz
    pico = np.max(np.abs(raw))
    if pico > 0: raw = raw / pico * 0.95
    return resamplear(raw, MIC_SR, TARGET_SR)


def salvar(audio, classe, numero):
    pasta = os.path.join(DATASET_DIR, classe); os.makedirs(pasta, exist_ok=True)
    cam = os.path.join(pasta, f"sample_{numero:03d}.wav")
    sf.write(cam, audio, TARGET_SR, subtype="PCM_16")
    return cam


def menu():
    print("\n" + "="*54)
    print("   GRAVADOR DE DATASET - Voice Accessibility AI")
    print("="*54)
    if MIC_DEVICE is not None:
        nome_mic = sd.query_devices(MIC_DEVICE)["name"]
        print(f"   Microfone: [{MIC_DEVICE}] {nome_mic} @ {MIC_SR}Hz")
    else:
        print("   AVISO: microfone nao detectado!")
    print()
    for num, cl in CLASSES.items():
        pasta = os.path.join(DATASET_DIR, cl)
        total = len(os.listdir(pasta)) if os.path.isdir(pasta) else 0
        frases = " / ".join(FRASES[cl][:2])
        print(f"   {num}. {cl:<20} ({total:>2} amostras)  {frases}")
    print("\n   0. Sair\n")
    return input("   Sua escolha: ").strip()


def sessao(classe):
    pasta = os.path.join(DATASET_DIR, classe); os.makedirs(pasta, exist_ok=True)
    frases = FRASES[classe]
    print(f"\n   Classe: {classe}")
    print(f"   Fale:   {' / '.join(frases)}")
    print(f"   Duracao: {TARGET_DURATION}s por amostra")
    print("   ENTER=gravar | q=menu | s=ouvir ultima\n")

    while True:
        numero = proximo_numero(pasta)
        entrada = input(f"   [{numero} salvas] > ").strip().lower()

        if entrada == "q":
            break

        if entrada == "s":
            if numero == 0:
                print("   Nenhuma amostra ainda."); continue
            d, sr = sf.read(os.path.join(pasta, f"sample_{(numero-1):03d}.wav"))
            print("   Reproduzindo..."); sd.play(d, sr); sd.wait(); continue

        frase = frases[numero % len(frases)]
        print(f"\n   Prepare-se para falar: \"{frase}\"")
        contar_regressiva(3)

        try:
            audio = gravar_audio()
        except Exception as e:
            print(f"   ERRO na gravacao: {e}"); continue

        cam = salvar(audio, classe, numero)
        db = 20 * np.log10(np.max(np.abs(audio)) + 1e-9)
        qualidade = "BOM" if db > -20 else "BAIXO - fale mais perto do microfone"
        print(f"   Salvo: {os.path.basename(cam)}")
        print(f"   Volume: {db:.1f} dB  [{qualidade}]\n")


def main():
    if MIC_DEVICE is None:
        print("\n   ERRO: nenhum microfone encontrado!"); return

    print(f"\n   Testando microfone [{MIC_DEVICE}]...", end=" ", flush=True)
    try:
        sd.rec(int(MIC_SR * 0.2), samplerate=MIC_SR, channels=1,
               dtype="float32", device=MIC_DEVICE)
        sd.wait()
        print("OK!")
    except Exception as e:
        print(f"\n   ERRO: {e}"); return

    while True:
        ch = menu()
        if ch == "0":
            print("\n   Gravacao encerrada. Ate logo!\n"); break
        elif ch in CLASSES:
            sessao(CLASSES[ch])
        else:
            print("   Opcao invalida.")


if __name__ == "__main__":
    main()
