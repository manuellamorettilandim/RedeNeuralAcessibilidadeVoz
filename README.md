# 🎙️ Voice Accessibility AI — Reconhecimento de Comandos de Voz por MFCC & Redes Neurais

> **Projeto Completo de Inteligência Artificial e Desenvolvimento Full-Stack focado em Acessibilidade Digital.**  
> Classificação de comandos de fala em tempo real utilizando **Coeficientes Cepstrais na Escala Mel (MFCCs)**, **Rede Neural Convolucional 2D (CNN)** e **Aplicação Web Interativa (FastAPI + HTML5/CSS/JavaScript)**.

---

## 🎯 Sumário Executivo do Projeto

O objetivo deste projeto é proporcionar **autonomia digital e inclusão** para pessoas com deficiências motoras (ex: tetraplegia, esclerose lateral amiotrófica — ELA, doença de Parkinson), baixa visão ou limitações temporárias (LER/DORT), através do reconhecimento de comandos curtos de voz executados instantaneamente no navegador:

| # | Comando Alvo | Ação de Acessibilidade Disparada |
|---|--------------|-----------------------------------|
| 1 | `"ler texto"` | Narração do conteúdo em voz alta via síntese vocal (Web Speech API) com marcação de leitura |
| 2 | `"aumentar fonte"` | Zoom tipográfico progressivo (+15% até 160% com reset automático) |
| 3 | `"mudar voz"` | Alternância dinâmica entre timbres, velocidades e perfis vocais com feedback sonoro |
| 4 | `"modo claro"` | Ativação do tema visual claro para ambientes com alta iluminação |
| 5 | `"modo escuro"` | Ativação do tema visual escuro com alto contraste WCAG AAA para conforto visual |
| 6 | `"background"` | Detecção de silêncio/ruído ambiente (elimina falsos disparos) |

---

## 🧠 ETAPA 1: Fundamentação Teórica

### 1. Redes Neurais Artificiais (RNAs)
As RNAs são modelos matemáticos inspirados na conectividade sináptica do cérebro humano.
- **Camadas (Layers):**
  - **Camada de Entrada (Input Layer):** Recebe o tensor tridimensional de features acústicas $(20 \times 47 \times 1)$, representando 20 coeficientes MFCC distribuídos em 47 janelas temporais de $32\text{ms}$.
  - **Camadas Ocultas (Hidden Layers):** Blocos convolucionais 2D (`Conv2D` com filtros $3 \times 3$) que funcionam como detectores locais de formantes e variações espectro-temporais, seguidos de `MaxPooling2D` para redução de dimensionalidade e `GlobalAveragePooling2D` para garantir invariância temporal.
  - **Camada de Saída (Output Layer):** Neurônios densos com ativação `Softmax`, que convertem os valores brutos (logits) em uma distribuição probabilística normalizada somando exatamente 100%:
    $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
- **Funções de Ativação:**
  - **ReLU ($f(x) = \max(0, x)$):** Introduz não-linearidade sem sofrer com o desaparecimento do gradiente (*vanishing gradient problem*), acelerando a convergência.

### 2. Aprendizado Supervisionado
O treinamento do modelo opera através de otimização supervisionada:
1. **Entrada de Features ($X$):** Matrizes MFCC normalizadas.
2. **Rótulos ($y$):** Classes codificadas em *One-Hot Encoding*.
3. **Função de Perda (Categorical Cross-Entropy):**
   $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
4. **Backpropagation:** Propagação reversa do erro aplicando a Regra da Cadeia do cálculo diferencial para determinar $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$.
5. **Otimizador Adam:** Ajuste adaptativo dos pesos com base nos momentos de 1ª e 2ª ordem dos gradientes, associado a *Learning Rate Decay* e *Early Stopping* para evitar *overfitting*.

### 3. MFCCs (Mel-Frequency Cepstral Coefficients)
O som cru (amplitudes no tempo) possui alta variabilidade de fase e volume. O ouvido humano percebe frequências de forma não-linear: temos altíssima sensibilidade a variações de tom em frequências graves ($< 1000\text{ Hz}$) e baixa sensibilidade em frequências agudas ($> 1000\text{ Hz}$).

A **Escala Mel** mapeia a percepção humana:
$$\text{Mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

#### O Pipeline de Extração MFCC (6 Passos):
```
[Sinal WAV 16kHz] 
       ↓
1. Pré-ênfase (y[t] = x[t] - α x[t-1])
       ↓
2. Janelamento de Hamming (Quadros de 32ms com hop de 512)
       ↓
3. Transformada Rápida de Fourier (FFT 1024)
       ↓
4. Banco de Filtros Triangulares Mel (Mel Filterbank)
       ↓
5. Logaritmo da Energia Espectral (Escala Decibel)
       ↓
6. Transformada Discreta do Cosseno (DCT)
       ↓
[Matriz MFCC 20 × 47] → Entrada da Rede Neural
```

---

## 📁 Estrutura do Repositório

```
c:/Users/25012000/Desktop/Nova pasta/
│
├── dataset/                         # Dataset de áudio balanceado (.wav)
│   ├── ler_texto/                   # Gravações do comando 1
│   ├── aumentar_fonte/              # Gravações do comando 2
│   ├── mudar_voz/                   # Gravações do comando 3
│   ├── alto_contraste/              # Gravações do comando 4
│   └── background/                  # Ruído térmico, silêncio e ar condicionado
│
├── models/                          # Artefatos do Modelo Treinado
│   ├── voice_command_model.keras    # Modelo CNN 2D Keras compilado
│   ├── labels.json                  # Mapeamento de classes e metadados
│   ├── metrics.json                 # Histórico completo de loss, acc e matriz de confusão
│   ├── training_history.png         # Gráfico exportado das curvas de aprendizado
│   └── confusion_matrix.png         # Gráfico exportado da matriz de confusão
│
├── static/                          # Frontend Moderno (HTML5, CSS Vanilla e JS)
│   ├── index.html                   # Aplicação Web SPA (5 abas interativas)
│   ├── css/
│   │   └── style.css                # Design system com Dark Mode, Glassmorphism e WCAG AAA
│   ├── js/
│   │   ├── app.js                   # Roteamento de abas, API REST e Canvas Heatmap
│   │   ├── audio_recorder.js        # Gravador Web Audio API e codificador WAV 16kHz
│   │   └── accessibility.js         # Simulador de acessibilidade (TTS, zoom, timbres)
│   └── demo_samples/                # Amostras para testes com 1 clique (sem microfone)
│       └── demo_manifest.json
│
├── dataset_prep.py                  # ETAPA 2: Pré-processamento, Extração MFCC e Síntese de Dados
├── train_model.py                   # ETAPA 3: Pipeline CNN Keras, Avaliação e Exportação
├── app.py                           # ETAPA 4 & 5: Backend FastAPI com inferência em tempo real
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Documentação completa
```

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python 3.10 ou superior
- Microfone conectado (opcional — o sistema possui bateria de testes rápidos embutida)

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Gerar / Pré-processar o Dataset (Opcional - já gerado)
```bash
python dataset_prep.py --samples 70
```
*Gera 350 amostras acústicas balanceadas com variações fonéticas e data augmentation.*

### 3. Treinar o Modelo Neural (Opcional - já treinado)
```bash
python train_model.py
```
*Treina a CNN 2D, salva os artefatos em `models/` e exporta os gráficos de avaliação.*

### 4. Iniciar o Servidor Web e Aplicação Interativa
```bash
python app.py
```
Abra o navegador no endereço: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🖥️ Estrutura das 5 Abas do Dashboard

1. **Aba 1: Demonstração Prática (Principal):**
   - Gravação com microfone em tempo real com osciloscópio visual em Canvas.
   - Upload de arquivos `.wav`.
   - Botões de teste rápido com 1 clique.
   - Exibição de comando previsto, confiança percentual e barras de probabilidade de todas as classes.
   - Heatmap espectro-temporal dos MFCCs extraídos.
   - **Simulador de Acessibilidade:** Execução em tempo real de narração de texto, zoom tipográfico, troca de voz e alto contraste.
2. **Aba 2: Teoria & Conceitos:**
   - Fundamentação matemática das RNAs, aprendizado supervisionado, escala Mel e as 6 etapas do MFCC.
3. **Aba 3: Pipeline de Dados:**
   - Explorador de amostras sonoras de cada comando, player de áudio e visualização espectral.
4. **Aba 4: Resultados & Métricas:**
   - Cards de KPI, curvas de aprendizado por época, matriz de confusão e relatório completo de precisão/revocação/F1-score.
5. **Aba 5: Conclusões & Futuro:**
   - Discussão crítica sobre ruído acústico, mitigação de falsos positivos, edge computing com TFLite/WebAssembly e impacto social.

---

## 📊 Métricas de Desempenho Alcançadas

| Métrica | Valor Obtido |
|---------|--------------|
| **Acurácia no Teste (Test Set)** | **85.71%** |
| **Acurácia Máxima de Validação** | **91.43%** |
| **Loss no Teste (Cross-Entropy)** | **0.3354** |
| **Latência Média de Inferência** | **~15 - 25 ms** |
| **Total de Parâmetros da CNN** | **245.125 parâmetros** |

---

## ♿ Conformidade de Acessibilidade
- **WCAG 2.2 Nível AAA:** Modo Alto Contraste dedicado em fundo preto absoluto (`#000000`) e elementos em amarelo vibrante (`#ffff00`).
- **Suporte a Leitores de Tela:** Estrutura HTML5 semântica e atributos ARIA (`aria-live="polite"`, `role="status"`).
- **Feedback Auditivo:** Bipes em frequências específicas para confirmação de comandos para usuários com baixa visão.
