"""
=============================================================================
PROJETO: Reconhecimento de Comandos de Voz para Acessibilidade
MÓDULO: train_model.py - Treinamento e Avaliação da Rede Neural Convolucional (PyTorch)
=============================================================================

Este script executa:
1. Carregamento dos dados e extração das features MFCC (20 x 47).
2. Divisão estratificada em Treino (80%) e Teste (20%).
3. Construção da Rede Neural Convolucional 2D (CNN) em PyTorch.
4. Treinamento com Regularização (BatchNorm, Dropout, Early Stopping / Best Checkpoint).
5. Avaliação do modelo (Acurácia global, Matriz de Confusão e Classification Report).
6. Exportação dos artefatos:
   - models/voice_command_model.pt
   - models/labels.json
   - models/metrics.json
   - models/training_history.png
   - models/confusion_matrix.png
=============================================================================
"""

import os
import glob
import json
import time
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from dataset_prep import AudioPreprocessor, TARGET_SR
from model_arch import VoiceCommandCNN

CLASS_METADATA = {
    "aumentar_fonte": {
        "display_name": "Aumentar Fonte",
        "action": "increase_font",
        "description": "Zoom tipográfico para baixa visão",
        "color": "#10b981"
    },
    "diminuir_fonte": {
        "display_name": "Diminuir Fonte",
        "action": "decrease_font",
        "description": "Redução do tamanho da fonte",
        "color": "#ec4899"
    },
    "ler_texto": {
        "display_name": "Ler Texto",
        "action": "read_text",
        "description": "Leitor de tela por voz sintetizada",
        "color": "#3b82f6"
    },
    "modo_claro": {
        "display_name": "Modo Claro",
        "action": "set_light_theme",
        "description": "Ativação do tema visual claro",
        "color": "#f59e0b"
    },
    "modo_escuro": {
        "display_name": "Modo Escuro",
        "action": "set_dark_theme",
        "description": "Ativação do tema visual escuro com alto contraste",
        "color": "#06b6d4"
    },
    "mudar_voz": {
        "display_name": "Mudar Voz",
        "action": "change_voice",
        "description": "Alternância de perfil e timbre vocal",
        "color": "#8b5cf6"
    },
    "background": {
        "display_name": "Nenhum Comando",
        "action": "none",
        "description": "Ruído ambiente, silêncio ou fala não relacionada",
        "color": "#64748b"
    }
}


def discover_classes(dataset_dir="dataset"):
    """
    Descobre todas as classes com arquivos .wav no diretório do dataset.
    """
    classes = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and len(glob.glob(os.path.join(dataset_dir, d, "*.wav"))) > 0
    ]
    classes.sort()
    return classes


def load_dataset(dataset_dir="dataset"):
    """
    Carrega todos os arquivos de áudio por classe e extrai as matrizes MFCC.
    Retorna X com shape (N, 1, 20, 47), y com rótulos inteiros e lista de classes.
    """
    classes = discover_classes(dataset_dir)
    if not classes:
        raise ValueError(f"Nenhuma classe com arquivos .wav encontrada em '{dataset_dir}'")

    preprocessor = AudioPreprocessor()
    X = []
    y = []
    filenames = []
    groups = []
    groups_seen = []
    rng = np.random.default_rng(42)

    print(f"\n[+] Carregando dados do dataset '{dataset_dir}' ({len(classes)} classes detectadas)...")
    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_dir, class_name)
        audio_files = glob.glob(os.path.join(class_folder, "*.wav"))
        print(f"    - Classe '{class_name}' ({class_idx}): {len(audio_files)} arquivos")

        for fpath in audio_files:
            try:
                group_id = len(groups_seen)
                groups_seen.append(fpath)
                wave = preprocessor.load_audio(fpath)
                X.append(preprocessor.extract_mfcc(wave))
                y.append(class_idx)
                groups.append(group_id)
                # Variantes que simulam microfone de navegador (ruído + deslocamento temporal)
                for _ in range(2):
                    noisy = np.roll(wave, int(rng.uniform(-0.3, 0.3) * TARGET_SR))
                    sig_pow = np.mean(noisy ** 2) + 1e-9
                    snr_db = rng.uniform(12, 28)
                    noise = rng.normal(0, np.sqrt(sig_pow / (10 ** (snr_db / 10))), len(noisy))
                    noisy = (noisy * rng.uniform(0.3, 1.2) + noise).astype(np.float32)
                    X.append(preprocessor.extract_mfcc(noisy))
                    y.append(class_idx)
                    groups.append(group_id)
            except Exception as e:
                print(f"      [!] Falha no arquivo {fpath}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups)

    # Adiciona dimensão de canal para convolução: (N, 1, 20, 47)
    X = np.expand_dims(X, axis=1)

    print(f"\n[OK] Dataset carregado com sucesso!")
    print(f"    - Formato de X: {X.shape} (N, Canais, Coef_MFCC, Janelas)")
    print(f"    - Formato de y: {y.shape}")
    print(f"    - Total de amostras (com augmentation): {len(y)}")

    return X, y, classes, groups


def train_pytorch_model(X_train, y_train, X_test, y_test, num_classes, epochs=35, batch_size=16, lr=0.001):
    """
    Treina a rede neural convolucional VoiceCommandCNN em PyTorch.
    """
    device = torch.device("cpu")
    print(f"\n[+] Inicializando treinamento no dispositivo: {device}")

    # Cria DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VoiceCommandCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    best_val_acc = 0.0
    best_weights = None

    print(f"[+] Iniciando {epochs} épocas de treinamento (batch_size={batch_size})...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Modo treino
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Modo avaliação
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_acc)

        history['loss'].append(round(epoch_loss, 4))
        history['accuracy'].append(round(epoch_acc, 4))
        history['val_loss'].append(round(epoch_val_loss, 4))
        history['val_accuracy'].append(round(epoch_val_acc, 4))

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"    Época [{epoch:02d}/{epochs:02d}] - "
                  f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.1f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}%")

    training_time = round(time.time() - start_time, 2)
    print(f"\n[OK] Treinamento concluído em {training_time}s! Melhor Val Acc: {best_val_acc*100:.1f}%")

    # Restaura melhores pesos
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history, training_time


def evaluate_and_plot(model, X_test, y_test, classes, history, output_dir="models"):
    """
    Avalia o modelo final, gera gráficos de aprendizado e matriz de confusão.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    device = torch.device("cpu")
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test).to(device)
        logits = model(test_tensor)
        probs = F.softmax(logits, dim=1).numpy()
        y_pred = np.argmax(probs, axis=1)

    # Métricas
    test_acc = float(np.mean(y_pred == y_test))
    criterion = nn.CrossEntropyLoss()
    test_loss = float(criterion(logits, torch.from_numpy(y_test)).item())

    print(f"\n=======================================================")
    print(f"RESULTADOS DA AVALIAÇÃO FINAL NO CONJUNTO DE TESTE:")
    print(f"=======================================================")
    print(f"Acurácia Global (Test Accuracy): {test_acc * 100:.2f}%")
    print(f"Perda de Teste (Test Loss):      {test_loss:.4f}")

    # Classification Report
    cls_report_dict = classification_report(
        y_test, y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    cls_report_text = classification_report(
        y_test, y_pred,
        target_names=classes,
        zero_division=0
    )
    print("\nRelatório de Classificação Detalhado:")
    print(cls_report_text)

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    # 1. Gráfico do Histórico de Treinamento
    history_plot_path = os.path.join(output_dir, "training_history.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs_range = range(1, len(history['loss']) + 1)
    # Acurácia
    axes[0].plot(epochs_range, [a * 100 for a in history['accuracy']], 'o-', label='Treino', color='#3b82f6', lw=2)
    axes[0].plot(epochs_range, [a * 100 for a in history['val_accuracy']], 's--', label='Validação', color='#10b981', lw=2)
    axes[0].set_title("Evolução da Acurácia (%)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("Acurácia (%)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Loss
    axes[1].plot(epochs_range, history['loss'], 'o-', label='Treino', color='#ef4444', lw=2)
    axes[1].plot(epochs_range, history['val_loss'], 's--', label='Validação', color='#f59e0b', lw=2)
    axes[1].set_title("Evolução da Perda (Cross-Entropy)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Épocas")
    axes[1].set_ylabel("Perda")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=150)
    plt.close()
    print(f"[OK] Gráfico de histórico salvo em: {history_plot_path}")

    # 2. Gráfico da Matriz de Confusão
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues, alpha=0.85)

    display_labels = [CLASS_METADATA.get(c, {}).get("display_name", c) for c in classes]

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center", color=color, fontsize=10, fontweight='bold')

    fig.colorbar(cax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(display_labels, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.set_xlabel("Predição da IA", fontweight="bold", labelpad=10)
    ax.set_ylabel("Classe Real Gravada", fontweight="bold")
    ax.set_title(f"Matriz de Confusão Normalizada (Acurácia: {test_acc*100:.1f}%)", fontsize=11, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"[OK] Matriz de confusão salva em: {cm_plot_path}")

    return {
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.round(3).tolist(),
        "classification_report": cls_report_dict
    }


def save_artifacts(model, classes, history, eval_metrics, training_time, output_dir="models"):
    """
    Exporta o modelo treinado em PyTorch, metadados e métricas JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Salva modelo PyTorch (.pt)
    model_path = os.path.join(output_dir, "voice_command_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': len(classes),
        'classes': classes,
        'input_shape': [1, 20, 47]
    }, model_path)
    print(f"[OK] Modelo PyTorch salvo em: {model_path}")

    # 2. Salva metadados dos rótulos (labels.json)
    labels_path = os.path.join(output_dir, "labels.json")
    labels_content = {
        "classes": classes,
        "input_shape": [1, 20, 47],
        "metadata": {c: CLASS_METADATA.get(c, {
            "display_name": c,
            "action": c,
            "description": f"Comando {c}",
            "color": "#64748b"
        }) for c in classes}
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_content, f, indent=2, ensure_ascii=False)
    print(f"[OK] Rótulos e metadados salvos em: {labels_path}")

    # 3. Salva métricas de treinamento (metrics.json)
    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics_content = {
        "timestamp": datetime.now().isoformat(),
        "test_accuracy": eval_metrics["test_accuracy"],
        "test_loss": eval_metrics["test_loss"],
        "training_time_sec": training_time,
        "epochs": len(history['loss']),
        "classes": classes,
        "history": history,
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "confusion_matrix_normalized": eval_metrics["confusion_matrix_normalized"],
        "classification_report": eval_metrics["classification_report"]
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_content, f, indent=2, ensure_ascii=False)
    print(f"[OK] Métricas salvas em: {metrics_path}")


def main():
    print("=" * 70)
    print("TREINAMENTO DA REDE NEURAL CONVOLUCIONAL 2D (PYTORCH) COM MFCC")
    print("=" * 70)

    # 1. Carrega dados
    X, y, classes, groups = load_dataset("dataset")

    # 2. Divisão 80/20 por arquivo de origem (variantes ruidosas ficam no mesmo lado)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\n[+] Divisão dos Dados:")
    print(f"    - Conjunto de Treino: {len(X_train)} amostras (80%)")
    print(f"    - Conjunto de Teste:  {len(X_test)} amostras (20%)")

    # 3. Treina modelo
    model, history, training_time = train_pytorch_model(
        X_train, y_train,
        X_test, y_test,
        num_classes=len(classes),
        epochs=35,
        batch_size=16,
        lr=0.001
    )

    # 4. Avalia e gera visualizações
    eval_metrics = evaluate_and_plot(model, X_test, y_test, classes, history)

    # 5. Salva todos os artefatos
    save_artifacts(model, classes, history, eval_metrics, training_time)

    print("\n" + "=" * 70)
    print("TREINAMENTO E EXPORTAÇÃO CONCLUÍDOS COM SUCESSO!")
    print(f"Acurácia no Conjunto de Teste: {eval_metrics['test_accuracy'] * 100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
