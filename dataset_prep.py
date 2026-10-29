"""
=============================================================================
PROJETO: Reconhecimento de Comandos de Voz para Acessibilidade
MÓDULO: dataset_prep.py - Coleta, Extração e Pré-processamento de Áudio (MFCC)
=============================================================================

Este script é responsável por:
1. Normalização temporal e amplitude de sinais sonoros (.wav).
2. Extração de características de MFCC (Mel-Frequency Cepstral Coefficients).
3. Data Augmentation (pitch-shift, time-stretch, adição de ruído e jitter temporal).
4. Geração automatizada de dataset balanceado de comandos de voz em pt-BR via SAPI
   para as classes:
     - "ler texto" (Comando 1)
     - "aumentar fonte" (Comando 2)
     - "mudar voz" (Comando 3)
     - "alto contraste" (Comando 4 - Acessibilidade visual)
     - "background" (Ruído ambiente / silêncio)
=============================================================================
"""

import os
import glob
import json
import random
import numpy as np
import soundfile as sf
import librosa

# Configurações globais de áudio
TARGET_SR = 16000          # Taxa de amostragem padrão (16 kHz - padrão ouro em ASR)
TARGET_DURATION = 1.5      # Duração fixa em segundos
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 24.000 amostras
N_MFCC = 20                # 20 coeficientes MFCC
N_FFT = 1024               # Tamanho da janela da FFT
HOP_LENGTH = 512           # Passo entre janelas consecutivas

CLASSES = [
    "aumentar_fonte",
    "diminuir_fonte",
    "ler_texto",
    "modo_claro",
    "modo_escuro",
    "mudar_voz",
    "background"   # Fix 4: classe de rejeição — ruído, silêncio, falas não-comando
]

PHRASES = {
    "aumentar_fonte": ["aumentar fonte", "aumentar a fonte", "aumenta a fonte", "aumenta fonte"],
    "diminuir_fonte": ["diminuir fonte", "diminuir a fonte", "diminui a fonte", "diminui fonte"],
    "ler_texto": ["ler texto", "ler o texto", "leia o texto", "ler este texto"],
    "modo_claro": ["modo claro", "ativar modo claro", "tema claro", "claro"],
    "modo_escuro": ["modo escuro", "ativar modo escuro", "tema escuro", "escuro"],
    "mudar_voz": ["mudar voz", "mudar a voz", "trocar voz", "troca a voz"]
}


class AudioPreprocessor:
    """
    Classe utilitária para tratamento de áudio e extração de MFCCs.
    """

    def __init__(self, target_sr=TARGET_SR, target_samples=TARGET_SAMPLES,
                 n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
        self.target_sr = target_sr
        self.target_samples = target_samples
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, file_path_or_bytes):
        """
        Carrega áudio de arquivo (.wav) ou buffer em memória, convertendo para mono
        e resampleando para a taxa de amostragem alvo.
        """
        try:
            y, sr = librosa.load(file_path_or_bytes, sr=self.target_sr, mono=True)
            return y
        except Exception as e:
            raise ValueError(f"Erro ao carregar áudio: {e}")

    def pad_or_trim(self, y):
        """
        Garante que o sinal de áudio tenha exatamente `target_samples`.
        Se for menor, aplica padding com zeros centralizado.
        Se for maior, trunca centralizando a energia principal do sinal.
        """
        if len(y) == 0:
            return np.zeros(self.target_samples, dtype=np.float32)

        if len(y) < self.target_samples:
            total_pad = self.target_samples - len(y)
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            return np.pad(y, (pad_left, pad_right), mode='constant')
        elif len(y) > self.target_samples:
            # Encontra o ponto de maior energia para não cortar o início da fala
            hop = 256
            frame_energy = np.array([
                np.sum(y[i:i + hop] ** 2) for i in range(0, len(y) - hop, hop)
            ])
            if len(frame_energy) > 0 and np.max(frame_energy) > 0:
                peak_idx = int(np.argmax(frame_energy) * hop)
                start = max(0, peak_idx - self.target_samples // 2)
                end = start + self.target_samples
                if end > len(y):
                    end = len(y)
                    start = max(0, end - self.target_samples)
                return y[start:end]
            return y[:self.target_samples]
        return y

    def normalize_amplitude(self, y):
        """
        Normaliza a amplitude do sinal pelo valor de pico para a faixa [-0.95, 0.95]
        quando há sinal de voz detectado (> 0.05). Sinais de ruído e silêncio
        são preservados sem amplificação artificial.
        """
        max_val = np.max(np.abs(y))
        if max_val > 0.05:
            return (y / max_val) * 0.95
        return y

    def extract_mfcc(self, y):
        """
        Extrai a matriz de coeficientes Mel-Frequency Cepstral (MFCC).
        Aplica padronização (z-score) global da matriz para preservar
        a dinâmica e as relações espectro-temporais entre frequências.
        
        Retorna matriz de formato (n_mfcc, time_frames), tipicamente (20, 47).
        """
        y_fixed = self.pad_or_trim(y)
        y_norm = self.normalize_amplitude(y_fixed)

        # Extração de MFCCs via Librosa
        mfcc = librosa.feature.mfcc(
            y=y_norm,
            sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Padronização z-score global
        mfcc_mean = np.mean(mfcc)
        mfcc_std = np.std(mfcc) + 1e-6
        mfcc_norm = (mfcc - mfcc_mean) / mfcc_std

        return mfcc_norm.astype(np.float32)

    def extract_mfcc_from_file(self, file_path):
        """
        Lê arquivo direto do disco e retorna a matriz MFCC normalizada.
        """
        y = self.load_audio(file_path)
        return self.extract_mfcc(y)

    def augment_audio(self, y, rng=None):
        """
        Aplica técnicas de Data Augmentation no sinal acústico:
        - Pitch Shift (alteração de tom / semitons)
        - Time Stretch (alongamento ou compressão temporal)
        - Adição de Ruído Gaussiano / Rosa (SNR variado)
        - Jitter / Deslocamento temporal aleatório
        - Variação de ganho
        """
        if rng is None:
            rng = random.Random()

        y_aug = y.copy()

        # 1. Jitter temporal (deslocamento leve)
        shift_amount = rng.randint(-int(self.target_sr * 0.15), int(self.target_sr * 0.15))
        y_aug = np.roll(y_aug, shift_amount)

        # 2. Variação de ganho aleatório
        gain = rng.uniform(0.7, 1.25)
        y_aug = y_aug * gain

        # 3. Time stretch aleatório
        if rng.random() > 0.4:
            rate = rng.uniform(0.85, 1.18)
            y_aug = librosa.effects.time_stretch(y_aug, rate=rate)

        # 4. Pitch shift aleatório (-3 a +3 semitons)
        if rng.random() > 0.4:
            n_steps = rng.uniform(-3.0, 3.0)
            y_aug = librosa.effects.pitch_shift(y_aug, sr=self.target_sr, n_steps=n_steps)

        # 5. Adição de ruído de fundo sutil
        if rng.random() > 0.3:
            noise_amp = rng.uniform(0.002, 0.02)
            noise = np.random.normal(0, noise_amp, len(y_aug)).astype(np.float32)
            y_aug = y_aug + noise

        return y_aug


def synthesize_speech_sapi(text, output_path, rate=0, volume=100, voice_name=None):
    """
    Sintetiza fala nativa via Windows SAPI COM em arquivo WAV.
    """
    try:
        import comtypes.client
        speaker = comtypes.client.CreateObject('SAPI.SpVoice')
        
        # Seleciona voz se especificada
        if voice_name:
            voices = speaker.GetVoices()
            for i in range(voices.Count):
                v = voices.Item(i)
                if voice_name.lower() in v.GetDescription().lower():
                    speaker.Voice = v
                    break
        
        speaker.Rate = int(rate)
        speaker.Volume = int(volume)

        filestream = comtypes.client.CreateObject('SAPI.SpFileStream')
        # 3 = SSFMCreateForWrite
        filestream.Open(output_path, 3, False)
        speaker.AudioOutputStream = filestream
        speaker.Speak(text)
        filestream.Close()
        return True
    except Exception as e:
        print(f"Erro na síntese SAPI para '{text}': {e}")
        return False


def generate_synthetic_noise(duration_sec=1.5, sr=16000, noise_type="white"):
    """
    Gera sinais acústicos de ruído / fundo / silêncio realistas.
    """
    samples = int(duration_sec * sr)
    if noise_type == "silence":
        # Silêncio com ruído térmico mínimo de sensor
        return np.random.normal(0, 0.001, samples).astype(np.float32)
    elif noise_type == "pink":
        # Ruído rosa (1/f) - simula ar condicionado e ventilação
        uneven = samples % 2
        X = np.random.randn(samples // 2 + 1 + uneven) + 1j * np.random.randn(samples // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)
        y = (np.fft.irfft(X / S)).real[:samples]
        y = y / (np.max(np.abs(y)) + 1e-6) * 0.05
        return y.astype(np.float32)
    elif noise_type == "ambient":
        # Simulação de ruído ambiente com cliques sutis (teclado / respiração)
        y = np.random.normal(0, 0.005, samples).astype(np.float32)
        # Injeta 1 a 3 pequenos cliques
        for _ in range(random.randint(1, 3)):
            pos = random.randint(0, samples - 200)
            click_len = random.randint(50, 150)
            y[pos:pos + click_len] += np.random.normal(0, 0.08, click_len)
        return y.astype(np.float32)
    else:
        # Ruído branco suave
        return np.random.normal(0, 0.02, samples).astype(np.float32)


def generate_full_dataset(dataset_dir="dataset", samples_per_class=60):
    """
    Gera um dataset balanceado, profissional e robusto para todas as 5 classes.
    Usa variações fonéticas em português, modulação de velocidade SAPI (-3 a +4),
    variações de volume e Data Augmentation (pitch, stretch, ruído).
    """
    print("=" * 70)
    print("INICIANDO GERAÇÃO DO DATASET DE COMANDOS DE VOZ")
    print(f"Diretório de destino: {dataset_dir}")
    print(f"Meta de amostras por classe: ~{samples_per_class}")
    print("=" * 70)

    preprocessor = AudioPreprocessor()
    os.makedirs(dataset_dir, exist_ok=True)

    # Identifica vozes disponíveis
    pt_voice = "Maria"
    en_voice = "Zira"

    temp_dir = os.path.join(dataset_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    total_generated = 0

    for cls in CLASSES:
        cls_dir = os.path.join(dataset_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        print(f"\n[+] Processando classe: '{cls}'...")

        if cls == "background":
            # Geração de ruído de fundo, silêncio e cliques
            noise_types = ["silence", "pink", "ambient", "white"]
            for idx in range(samples_per_class):
                nt = noise_types[idx % len(noise_types)]
                duration = random.uniform(1.2, 1.8)
                audio = generate_synthetic_noise(duration_sec=duration, sr=TARGET_SR, noise_type=nt)
                audio_fixed = preprocessor.pad_or_trim(audio)
                out_path = os.path.join(cls_dir, f"sample_{idx:03d}.wav")
                sf.write(out_path, audio_fixed, TARGET_SR)
                total_generated += 1
            print(f"    -> {samples_per_class} amostras de background geradas com sucesso.")
            continue

        # Classes de comando de voz
        phrases = PHRASES[cls]
        rates = [-4, -2, -1, 0, 1, 2, 3, 4]
        volumes = [80, 90, 100]
        voices = [pt_voice, en_voice]

        base_audios = []
        base_count = 0

        # Passo 1: Síntese SAPI das variações base
        for phrase in phrases:
            for rate in rates:
                for voice in voices:
                    temp_file = os.path.join(temp_dir, f"base_{cls}_{base_count}.wav")
                    vol = random.choice(volumes)
                    ok = synthesize_speech_sapi(phrase, temp_file, rate=rate, volume=vol, voice_name=voice)
                    if ok and os.path.exists(temp_file):
                        try:
                            y, _ = librosa.load(temp_file, sr=TARGET_SR)
                            if len(y) > int(TARGET_SR * 0.2):  # Pelo menos 200ms
                                base_audios.append(y)
                                base_count += 1
                        except Exception:
                            pass

        if len(base_audios) == 0:
            print(f"[!] AVISO: Não foi possível sintetizar base via SAPI para '{cls}'. Gerando sinal senoidal de fallback.")
            base_audios = [np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000)).astype(np.float32)]

        print(f"    -> {len(base_audios)} gravações base sintetizadas.")

        # Passo 2: Geração do conjunto final com Data Augmentation
        sample_idx = 0
        rng = random.Random(42)

        # Salva primeiro as gravações base limpas
        for base_y in base_audios:
            if sample_idx >= samples_per_class:
                break
            y_fixed = preprocessor.pad_or_trim(base_y)
            y_norm = preprocessor.normalize_amplitude(y_fixed)
            out_path = os.path.join(cls_dir, f"sample_{sample_idx:03d}.wav")
            sf.write(out_path, y_norm, TARGET_SR)
            sample_idx += 1
            total_generated += 1

        # Completa com amostras aumentadas até atingir a meta
        while sample_idx < samples_per_class:
            base_choice = rng.choice(base_audios)
            y_aug = preprocessor.augment_audio(base_choice, rng=rng)
            y_fixed = preprocessor.pad_or_trim(y_aug)
            y_norm = preprocessor.normalize_amplitude(y_fixed)
            out_path = os.path.join(cls_dir, f"sample_{sample_idx:03d}.wav")
            sf.write(out_path, y_norm, TARGET_SR)
            sample_idx += 1
            total_generated += 1

        print(f"    -> Total de {sample_idx} arquivos gravados em '{cls_dir}'.")

    # Limpeza de temporários
    for f in glob.glob(os.path.join(temp_dir, "*.wav")):
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    # Cria amostras de demonstração para a Web App
    create_web_demo_samples(dataset_dir)

    print("\n" + "=" * 70)
    print(f"DATASET CONCLUÍDO COM SUCESSO! Total de arquivos: {total_generated}")
    print("=" * 70)
    return total_generated


def create_web_demo_samples(dataset_dir, output_dir="static/demo_samples"):
    """
    Copia 1 amostra de alta qualidade de cada classe para `static/demo_samples/`
    e gera um manifesto JSON para o frontend carregar nos botões de teste rápido.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    friendly_names = {
        "ler_texto": "Ler Texto (Comando 1)",
        "aumentar_fonte": "Aumentar Fonte (Comando 2)",
        "mudar_voz": "Mudar Voz (Comando 3)",
        "modo_claro": "Modo Claro (Comando 4)",
        "modo_escuro": "Modo Escuro (Comando 5)",
        "alto_contraste": "Modo Claro / Escuro",
        "background": "Ruído Ambiente / Silêncio"
    }

    descriptions = {
        "ler_texto": "Dispara o leitor de tela para narrar o conteúdo acessível",
        "aumentar_fonte": "Aumenta a escala tipográfica da interface para baixa visão",
        "mudar_voz": "Alterna entre diferentes perfis vocais e velocidades de fala",
        "modo_claro": "Ativa o tema visual claro para ambientes iluminados",
        "modo_escuro": "Ativa o tema visual escuro para descanso visual",
        "alto_contraste": "Alterna a interface visual entre tema claro e tema escuro",
        "background": "Áudio sem comando (simula ruído de fundo ou silêncio)"
    }

    for cls in CLASSES:
        cls_path = os.path.join(dataset_dir, cls)
        samples = glob.glob(os.path.join(cls_path, "*.wav"))
        if samples:
            # Escolhe o primeiro arquivo limpo
            src_file = samples[0]
            dest_filename = f"demo_{cls}.wav"
            dest_path = os.path.join(output_dir, dest_filename)
            
            # Carrega e re-salva para garantir padronização
            y, sr = librosa.load(src_file, sr=TARGET_SR)
            sf.write(dest_path, y, sr)

            manifest.append({
                "class_id": cls,
                "label": friendly_names.get(cls, cls),
                "description": descriptions.get(cls, ""),
                "filename": dest_filename,
                "url": f"/static/demo_samples/{dest_filename}"
            })

    manifest_path = os.path.join(output_dir, "demo_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[+] Manifesto de demonstração salvo em: {manifest_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pré-processamento e Geração do Dataset de Comandos de Voz")
    parser.add_argument("--samples", type=int, default=60, help="Quantidade de amostras por classe")
    parser.add_argument("--dir", type=str, default="dataset", help="Diretório do dataset")
    args = parser.parse_args()

    generate_full_dataset(dataset_dir=args.dir, samples_per_class=args.samples)
