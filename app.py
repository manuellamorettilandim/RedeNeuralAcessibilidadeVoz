"""
=============================================================================
PROJETO: Reconhecimento de Comandos de Voz para Acessibilidade
MÓDULO: app.py - Backend FastAPI com Inferência em Tempo Real (PyTorch)
=============================================================================

Endpoints da API:
- GET  /                     -> Serve a aplicação Web (SPA)
- POST /api/predict          -> Inferência de áudio WAV via CNN PyTorch + MFCC
- GET  /api/metrics          -> Métricas de treinamento, matriz de confusão e loss
- GET  /api/samples          -> Lista amostras pré-gravadas de demonstração
- GET  /api/model-info       -> Informações da arquitetura da rede e labels
=============================================================================
"""

import os
import io
import json
import time
import tempfile
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn.functional as F

from dataset_prep import AudioPreprocessor, TARGET_SR
from model_arch import VoiceCommandCNN

app = FastAPI(
    title="Voice Accessibility AI",
    description="Sistema de Reconhecimento de Comandos de Voz para Acessibilidade com MFCC e Redes Neurais",
    version="1.0.0"
)

# Habilita CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialização global do preprocessador e modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

preprocessor = AudioPreprocessor()
model = None
labels_data = None
metrics_data = None
is_pytorch = True

# Reconhecimento de fala offline (Whisper) - reserva quando o navegador não entende
whisper_model = None


def _load_whisper_background():
    global whisper_model
    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("[OK] Whisper (reconhecimento de fala offline) carregado.")
    except Exception as e:
        print(f"[!] Whisper indisponível: {e}")


def transcribe_audio(wave):
    """Transcreve (pt) o áudio 16 kHz com Whisper. Retorna '' se indisponível ou silêncio."""
    if whisper_model is None or wave is None or len(wave) == 0:
        return ""
    if float(np.max(np.abs(wave))) < 0.02:
        return ""
    try:
        peak = float(np.max(np.abs(wave)))
        w = (wave / peak * 0.9).astype(np.float32)
        segments, _info = whisper_model.transcribe(w, language="pt", beam_size=1, temperature=0.0,
                                                   condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        print(f"[!] Erro no Whisper: {e}")
        return ""


import threading
threading.Thread(target=_load_whisper_background, daemon=True).start()


def load_model_and_metadata():
    """
    Carrega o modelo treinado (PyTorch ou Keras) e os arquivos de metadados.
    """
    global model, labels_data, metrics_data, is_pytorch

    pt_model_path = os.path.join(MODELS_DIR, "voice_command_model.pt")
    keras_model_path = os.path.join(MODELS_DIR, "voice_command_model.keras")
    labels_path = os.path.join(MODELS_DIR, "labels.json")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    # 1. Carrega labels primeiro para saber classes caso necessário
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
            print(f"[OK] Rótulos carregados ({len(labels_data.get('classes', []))} classes)")
        except Exception as e:
            print(f"[!] Erro ao carregar labels.json: {e}")

    # 2. Carrega modelo PyTorch (.pt)
    if os.path.exists(pt_model_path):
        try:
            checkpoint = torch.load(pt_model_path, map_location=torch.device('cpu'))
            num_classes = checkpoint.get("num_classes", len(labels_data.get("classes", [])) if labels_data else 6)
            loaded_model = VoiceCommandCNN(num_classes=num_classes)
            if "model_state_dict" in checkpoint:
                loaded_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                loaded_model.load_state_dict(checkpoint)
            loaded_model.eval()
            model = loaded_model
            is_pytorch = True
            print(f"[OK] Modelo neural PyTorch carregado com sucesso de: {pt_model_path}")
        except Exception as e:
            print(f"[!] Erro ao carregar modelo PyTorch: {e}")

    # Fallback para Keras se pt não estiver disponível
    elif os.path.exists(keras_model_path):
        try:
            from tensorflow import keras
            model = keras.models.load_model(keras_model_path)
            is_pytorch = False
            print(f"[OK] Modelo neural Keras carregado de: {keras_model_path}")
        except Exception as e:
            print(f"[!] Erro ao carregar modelo Keras fallback: {e}")

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            print(f"[OK] Métricas de treinamento carregadas (Acurácia: {metrics_data.get('test_accuracy', 0)*100:.1f}%)")
        except Exception as e:
            print(f"[!] Erro ao carregar metrics.json: {e}")


# Carrega na inicialização
load_model_and_metadata()


# Monta arquivos estáticos
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")
app.mount("/css", StaticFiles(directory=os.path.join(STATIC_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(STATIC_DIR, "js")), name="js")
app.mount("/demo_samples", StaticFiles(directory=os.path.join(STATIC_DIR, "demo_samples")), name="demo_samples")


@app.get("/")
def serve_index():
    """
    Serve a página principal da aplicação Web interativa.
    """
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Voice Accessibility AI Backend Online. Interface em /static/index.html"}


@app.get("/api/model-info")
def get_model_info():
    """
    Retorna metadados do modelo e configuração dos comandos.
    """
    if not labels_data:
        raise HTTPException(status_code=503, detail="Metadados do modelo não carregados.")
    return {
        "classes": labels_data.get("classes", []),
        "metadata": labels_data.get("metadata", {}),
        "input_shape": labels_data.get("input_shape", [1, 20, 47]),
        "sample_rate": TARGET_SR,
        "n_mfcc": preprocessor.n_mfcc,
        "framework": "PyTorch" if is_pytorch else "TensorFlow/Keras"
    }


@app.get("/api/metrics")
def get_metrics():
    """
    Retorna as métricas e histórico de treinamento do modelo.
    """
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if metrics_data:
        return metrics_data
    raise HTTPException(status_code=404, detail="Arquivo de métricas não encontrado. Execute train_model.py primeiro.")


@app.get("/api/samples")
def get_demo_samples():
    """
    Retorna o manifesto de áudios de teste pré-carregados.
    """
    manifest_path = os.path.join(STATIC_DIR, "demo_samples", "demo_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.post("/api/predict")
async def predict_audio(audio: UploadFile = File(...)):
    """
    Recebe um arquivo de áudio WAV (do microfone ou upload),
    extrai os MFCCs e realiza a inferência com a CNN.
    """
    global model, labels_data, is_pytorch

    if model is None:
        load_model_and_metadata()
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo neural não inicializado.")

    start_time = time.time()

    # Lê os bytes do arquivo enviado
    try:
        content = await audio.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Arquivo de áudio vazio.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler áudio: {e}")

    # Salva temporariamente para leitura segura pelo librosa/soundfile
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        # Pré-processamento e Extração de MFCC
        try:
            y, sr = librosa.load(temp_path, sr=TARGET_SR, mono=True)
        except Exception:
            data, sr_orig = sf.read(temp_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            y = librosa.resample(data.astype(np.float32), orig_sr=sr_orig, target_sr=TARGET_SR)

        # Extrai MFCC: shape (20, 47)
        mfcc = preprocessor.extract_mfcc(y)

        # Inferência
        if is_pytorch:
            # Entrada PyTorch: (Batch=1, Channels=1, MFCC=20, Janelas=47)
            input_tensor = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                logits = model(input_tensor)
                predictions = F.softmax(logits, dim=1)[0].numpy()
        else:
            # Entrada Keras: (1, 20, 47, 1)
            input_tensor = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1)
            predictions = model.predict(input_tensor, verbose=0)[0]

        # Extrai predição máxima e probabilidades
        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[pred_idx])

        classes = labels_data.get("classes", []) if labels_data else [
            "aumentar_fonte", "diminuir_fonte", "ler_texto", "modo_claro", "modo_escuro", "mudar_voz"
        ]
        class_metadata = labels_data.get("metadata", {}) if labels_data else {}

        pred_class = classes[pred_idx] if pred_idx < len(classes) else "desconhecido"

        # Fix 4-B: se o modelo previu 'background', trata como ausência de comando
        if pred_class == "background":
            meta = {
                "display_name": "Nenhum Comando",
                "action": "none",
                "description": "Ruído de fundo ou silêncio detectado — nenhum comando reconhecido.",
                "color": "#64748b"
            }
        else:
            meta = class_metadata.get(pred_class, {})

        # Formata probabilidades de todas as classes
        probabilities = {}
        for idx, cls_name in enumerate(classes):
            cls_meta = class_metadata.get(cls_name, {})
            prob_val = float(predictions[idx]) if idx < len(predictions) else 0.0
            probabilities[cls_name] = {
                "display_name": cls_meta.get("display_name", cls_name),
                "probability": prob_val,
                "percentage": round(prob_val * 100, 1),
                "action": cls_meta.get("action", "none"),
                "color": cls_meta.get("color", "#64748b")
            }

        # Preview downsampleado do MFCC para renderização no Canvas
        mfcc_preview = mfcc.round(3).tolist()
        latency_ms = round((time.time() - start_time) * 1000, 2)
        transcript = transcribe_audio(y)

        return JSONResponse({
            "success": True,
            "transcript": transcript,
            "prediction": {
                "class_id": pred_class,
                "display_name": meta.get("display_name", pred_class),
                "confidence": round(confidence * 100, 1),
                "raw_confidence": confidence,
                "action": meta.get("action", "none"),
                "description": meta.get("description", ""),
                "color": meta.get("color", "#3b82f6")
            },
            "probabilities": probabilities,
            "mfcc_shape": list(mfcc.shape),
            "mfcc_preview": mfcc_preview,
            "audio_duration_sec": round(len(y) / TARGET_SR, 2),
            "latency_ms": latency_ms
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro durante inferência: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

