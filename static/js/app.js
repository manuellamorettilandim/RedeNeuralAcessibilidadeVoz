/**
 * =============================================================================
 * PROJETO: Reconhecimento de Comandos de Voz para Acessibilidade
 * MÓDULO: app.js - Controlador Principal da Aplicação Web SPA
 * =============================================================================
 */

// Se a página foi aberta direto do disco (file://), a API não existe: vai para o servidor.
if (window.location.protocol === "file:") {
  window.location.replace("http://127.0.0.1:8080/");
}

/**
 * Mapeia o texto reconhecido pelo navegador (Web Speech API, pt-BR) para um comando.
 */
const COMMAND_ACTIONS = {
  ler_texto: { action: "read_text", name: "Ler Texto" },
  aumentar_fonte: { action: "increase_font", name: "Aumentar Fonte" },
  diminuir_fonte: { action: "decrease_font", name: "Diminuir Fonte" },
  mudar_voz: { action: "change_voice", name: "Mudar Voz" },
  modo_claro: { action: "set_light_theme", name: "Modo Claro" },
  modo_escuro: { action: "set_dark_theme", name: "Modo Escuro" }
};

function matchCommandFromText(text) {
  // Fix 1: Regexes mais precisas — do mais específico ao mais genérico.
  // Fonte verificada ANTES de mudar_voz para evitar "mudar fonte" -> mudar_voz.
  const t = (text || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  if (!t.trim()) return null;

  // 1. Diminuir fonte
  if (/\bdiminu|\breduz\b|\bmenor\b|\bdecresc/.test(t)) return "diminuir_fonte";

  // 2. Aumentar fonte (antes de mudar_voz para "mudar fonte" não virar mudar_voz)
  if (/\baument|\bamplia|\bmaior\b/.test(t)) return "aumentar_fonte";

  // 3. Modo escuro — contexto primeiro, palavra isolada depois
  if (/modo\s+escur|tema\s+escur|\bescuro\b|\bdark\b/.test(t)) return "modo_escuro";

  // 4. Modo claro — exige contexto ou palavra exata (evita falso positivo em "declaração")
  if (/modo\s+clar|tema\s+clar|\bclaro\b|\bbranco\b/.test(t)) return "modo_claro";

  // 5. Mudar voz — agora exige "voz" explicitamente
  if (/\bvoz\b|vozes/.test(t)) return "mudar_voz";

  // 6. Ler texto
  if (/\bler\b|\bleia\b|leitura/.test(t)) return "ler_texto";

  // 7. Fonte genérico como último recurso -> aumentar
  if (/\bfonte\b|\btamanho\b|\bzoom\b/.test(t)) return "aumentar_fonte";

  return null;
}

/**
 * Escuta a fala (reconhecimento do navegador) durante a gravação. Retorna Promise<string>.
 */
function listenSpeech(ms) {
  return new Promise((resolve) => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return resolve("");
    let finalText = "";
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(finalText); } };
    try {
      const rec = new SR();
      rec.lang = "pt-BR";
      rec.interimResults = true;
      rec.maxAlternatives = 3;
      rec.continuous = false;
      rec.onresult = (e) => {
        let txt = "";
        for (let i = 0; i < e.results.length; i++) {
          for (let j = 0; j < e.results[i].length; j++) txt += " " + e.results[i][j].transcript;
        }
        finalText = txt;
      };
      rec.onerror = (ev) => { window.lastSpeechError = ev && ev.error ? ev.error : "erro"; finish(); };
      rec.onend = finish;
      rec.start();
      setTimeout(() => { try { rec.stop(); } catch (_) {} setTimeout(finish, 700); }, ms);
    } catch (_) { finish(); }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTabsNavigation();
  initAudioControls();
  loadQuickDemoSamples();
  loadMetricsAndResults();
  initAccessibilityButtons();
  initTextEditorControls();
  initZoomControls();
  initVoiceFontModalEvents();
});

/**
 * Navegação entre as 5 Abas do Dashboard
 */
function initTabsNavigation() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTabId = btn.getAttribute("data-tab");

      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(targetTabId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

/**
 * Controles de Áudio (Gravação Microfone e Upload de Arquivo)
 */
function initAudioControls() {
  const micBtn = document.getElementById("micRecordBtn");
  const recordStatus = document.getElementById("recordStatusText");
  const countdownEl = document.getElementById("countdownTimer");
  const waveCanvas = document.getElementById("liveWaveCanvas");
  const fileInput = document.getElementById("audioFileInput");

  if (!micBtn) return;

  // 1. Gravação ao vivo por microfone
  micBtn.addEventListener("click", async () => {
    if (window.audioRecorder.isRecording) return;

    micBtn.classList.add("recording");
    recordStatus.innerText = "Gravando... Fale o comando agora!";
    countdownEl.classList.add("active");

    const RECORD_SECONDS = 3;
    const speechPromise = listenSpeech(RECORD_SECONDS * 1000);

    window.audioRecorder.startRecording(
      RECORD_SECONDS,
      waveCanvas,
      (remaining) => {
        countdownEl.innerText = `${remaining}s`;
      },
      async (wavBlob) => {
        micBtn.classList.remove("recording");
        countdownEl.classList.remove("active");
        recordStatus.innerText = "Processando áudio com a Rede Neural...";

        const transcript = await speechPromise;
        await sendAudioForPrediction(wavBlob, "microphone_recording.wav", transcript);
      },
      (error) => {
        micBtn.classList.remove("recording");
        countdownEl.classList.remove("active");
        recordStatus.innerText = "Erro ao acessar microfone.";
        window.accessibilitySim.showToast(`Erro no microfone: ${error.message}`, "rose");
      }
    );
  });

  // 2. Upload manual de arquivo WAV
  if (fileInput) {
    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      recordStatus.innerText = `Enviando arquivo: ${file.name}...`;
      await sendAudioForPrediction(file, file.name);
      fileInput.value = "";
    });
  }
}

/**
 * Envia o áudio (Blob ou File) para o endpoint /api/predict
 */
async function sendAudioForPrediction(audioBlobOrFile, filename = "audio.wav", transcript = "") {
  const loadingOverlay = document.getElementById("predictionLoading");
  if (loadingOverlay) loadingOverlay.style.display = "block";

  const formData = new FormData();
  formData.append("audio", audioBlobOrFile, filename);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Erro na inferência do servidor.");
    }

    const data = await res.json();

    // Fix 2: Quando a fala do usuário reconhece um comando válido, ela tem prioridade absoluta
    if (!transcript && data.transcript) transcript = data.transcript;
    const spoken = matchCommandFromText(transcript);
    if (spoken && COMMAND_ACTIONS[spoken]) {
      data.prediction.class_id = spoken;
      data.prediction.display_name = COMMAND_ACTIONS[spoken].name;
      data.prediction.action = COMMAND_ACTIONS[spoken].action;
      data.prediction.confidence = Math.max(data.prediction.confidence, 95);
    }
    const statusEl = document.getElementById("recordStatusText");
    if (statusEl && transcript !== "") {
      statusEl.innerText = `Ouvi: "${transcript.trim()}" → ${data.prediction.display_name}`;
    } else if (statusEl && transcript === "" && filename === "microphone_recording.wav") {
      statusEl.innerText = `Não ouvi nada (${window.lastSpeechError || "sem fala"}). Usando só a CNN.`;
    }
    console.log("Fala reconhecida:", transcript, "->", spoken, "| CNN:", data.prediction.class_id);

    // Fix 3: Threshold elevado para 70% + bloqueio de classe 'background'.
    const MIN_CONFIDENCE = 70;

    // Bloqueia predição de ruído/background — não executa nenhuma ação
    if (data.prediction.class_id === "background" || data.prediction.action === "background") {
      displayPredictionResult(data);
      window.accessibilitySim.showToast("Nenhum comando identificado. Fale mais perto do microfone.", "amber");
      return;
    }

    // Sem fala reconhecida e CNN insegura: não dispara nada (evita ação errada)
    if (!spoken && data.prediction.confidence < MIN_CONFIDENCE) {
      displayPredictionResult(data);
      window.accessibilitySim.showToast(`Confiança baixa (${data.prediction.confidence}%). Fale o comando novamente.`, "amber");
      return;
    }

    displayPredictionResult(data);

    // Executa a ação de acessibilidade automaticamente!
    if (data.prediction && data.prediction.action && data.prediction.action !== "none") {
      window.accessibilitySim.showToast(`Comando: ${data.prediction.display_name}${transcript ? ' ("' + transcript.trim().split('  ')[0] + '")' : ''}`, "blue");
      window.accessibilitySim.executeAction(data.prediction.action);
    }

  } catch (error) {
    console.error("Erro na inferência:", error);
    window.accessibilitySim.showToast(`Erro na inferência: ${error.message}`, "rose");
  } finally {
    if (loadingOverlay) loadingOverlay.style.display = "none";
  }
}

/**
 * Exibe os resultados da inferência no painel da Aba 3
 */
function displayPredictionResult(data) {
  const pred = data.prediction;
  const probs = data.probabilities;

  // Atualiza destaque do comando previsto
  const predNameEl = document.getElementById("predClassName");
  const confidenceEl = document.getElementById("predConfidence");
  const latencyEl = document.getElementById("inferenceLatency");

  if (predNameEl) predNameEl.innerText = pred.display_name;
  if (confidenceEl) confidenceEl.innerText = `${pred.confidence}%`;
  if (latencyEl) latencyEl.innerText = `${data.latency_ms} ms`;

  // Atualiza barras de probabilidade
  const probListContainer = document.getElementById("probabilitiesList");
  if (probListContainer && probs) {
    probListContainer.innerHTML = "";

    Object.entries(probs).forEach(([classId, info]) => {
      const isWinner = classId === pred.class_id;
      const item = document.createElement("div");
      item.className = "prob-item";

      item.innerHTML = `
        <div class="prob-labels">
          <span style="color: ${isWinner ? 'var(--accent-cyan)' : 'var(--text-main)'}; font-weight: ${isWinner ? '700' : '500'}">
            ${isWinner ? '★ ' : ''}${info.display_name}
          </span>
          <span class="prob-val">${info.percentage}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${info.percentage}%; background: ${info.color || 'var(--accent-cyan)'}"></div>
        </div>
      `;
      probListContainer.appendChild(item);
    });
  }

  // Renderiza o mapa de calor MFCC no Canvas
  if (data.mfcc_preview) {
    renderMFCCHeatmap(data.mfcc_preview, "liveMfccCanvas");
  }
}

/**
 * Renderiza matriz MFCC em formato de mapa de calor em um Canvas
 */
function renderMFCCHeatmap(mfccMatrix, canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !mfccMatrix || mfccMatrix.length === 0) return;

  const ctx = canvas.getContext("2d");
  const numRows = mfccMatrix.length;      // 20 coeficientes MFCC
  const numCols = mfccMatrix[0].length;   // ~47 time frames

  canvas.width = canvas.parentElement.clientWidth || 400;
  canvas.height = 160;

  const cellWidth = canvas.width / numCols;
  const cellHeight = canvas.height / numRows;

  // Encontra valores min/max para mapeamento de cor
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      const v = mfccMatrix[r][c];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }

  const range = (maxVal - minVal) || 1;

  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < numCols; c++) {
      // Inverte linha para frequências baixas ficarem na base
      const drawRow = numRows - 1 - r;
      const normalized = (mfccMatrix[r][c] - minVal) / range;
      ctx.fillStyle = getHeatmapColor(normalized);
      ctx.fillRect(c * cellWidth, drawRow * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
    }
  }
}

/**
 * Gradiente de cores para visualização de espectrograma
 * (Azul escuro -> Ciano -> Amarelo -> Vermelho brilhante)
 */
function getHeatmapColor(val) {
  const v = Math.max(0, Math.min(1, val));
  if (v < 0.25) {
    // Azul escuro para ciano
    const t = v / 0.25;
    return `rgb(${Math.round(15 + t * 40)}, ${Math.round(23 + t * 160)}, ${Math.round(80 + t * 175)})`;
  } else if (v < 0.6) {
    // Ciano para amarelo/âmbar
    const t = (v - 0.25) / 0.35;
    return `rgb(${Math.round(55 + t * 190)}, ${Math.round(183 + t * 72)}, ${Math.round(255 - t * 240)})`;
  } else {
    // Amarelo para vermelho intenso
    const t = (v - 0.6) / 0.4;
    return `rgb(${Math.round(245 + t * 10)}, ${Math.round(255 - t * 200)}, ${Math.round(15 - t * 15)})`;
  }
}

/**
 * Carrega a bateria de amostras pré-gravadas para testes rápidos
 */
async function loadQuickDemoSamples() {
  const container = document.getElementById("quickSamplesList");
  const pipelineClassList = document.getElementById("pipelineClassList");

  try {
    const res = await fetch("/api/samples");
    if (!res.ok) return;
    const samples = await res.json();

    if (container && samples.length > 0) {
      container.innerHTML = "";
      samples.forEach(sample => {
        const btn = document.createElement("button");
        btn.className = "sample-test-btn";
        btn.innerHTML = `
          <strong>${sample.label}</strong>
          <span>${sample.description}</span>
        `;
        btn.addEventListener("click", async () => {
          window.accessibilitySim.showToast(`Carregando amostra: ${sample.label}...`, "cyan");
          const audioRes = await fetch(sample.url);
          const blob = await audioRes.blob();
          await sendAudioForPrediction(blob, sample.filename);
        });
        container.appendChild(btn);
      });
    }

    // Preenche lista de classes na Aba 2 (Pipeline de Dados)
    if (pipelineClassList && samples.length > 0) {
      pipelineClassList.innerHTML = "";
      samples.forEach((sample, idx) => {
        const item = document.createElement("div");
        item.className = `class-item-card ${idx === 0 ? 'active' : ''}`;
        item.innerHTML = `
          <div class="class-item-info">
            <h4>${sample.label}</h4>
            <span>${sample.description}</span>
          </div>
          <span style="font-size: 1.2rem;">▶</span>
        `;
        item.addEventListener("click", () => {
          document.querySelectorAll(".class-item-card").forEach(c => c.classList.remove("active"));
          item.classList.add("active");
          playPipelineSample(sample);
        });
        pipelineClassList.appendChild(item);
      });

      // Toca a primeira amostra por padrão
      playPipelineSample(samples[0]);
    }

  } catch (e) {
    console.error("Falha ao carregar amostras de demo:", e);
  }
}

/**
 * Toca e visualiza uma amostra na Aba 2
 */
function playPipelineSample(sample) {
  const audioEl = document.getElementById("pipelineAudioPlayer");
  const titleEl = document.getElementById("pipelineSampleTitle");

  if (titleEl) titleEl.innerText = sample.label;
  if (audioEl) {
    audioEl.src = sample.url;
  }

  // Busca inferência para exibir o MFCC correspondente na Aba 2
  fetch(sample.url)
    .then(r => r.blob())
    .then(b => {
      const fd = new FormData();
      fd.append("audio", b, sample.filename);
      return fetch("/api/predict", { method: "POST", body: fd });
    })
    .then(r => r.json())
    .then(data => {
      if (data.mfcc_preview) {
        renderMFCCHeatmap(data.mfcc_preview, "pipelineMfccCanvas");
      }
    })
    .catch(err => console.log("Erro ao carregar preview MFCC:", err));
}

/**
 * Carrega métricas e preenche a Aba 4
 */
async function loadMetricsAndResults() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) return;
    const metrics = await res.json();

    // Atualiza KPIs
    const testAccEl = document.getElementById("kpiTestAcc");
    const testLossEl = document.getElementById("kpiTestLoss");
    const totalSamplesEl = document.getElementById("kpiTotalSamples");
    const epochsEl = document.getElementById("kpiEpochs");

    if (testAccEl) testAccEl.innerText = `${(metrics.test_accuracy * 100).toFixed(1)}%`;
    if (testLossEl) testLossEl.innerText = metrics.test_loss.toFixed(4);
    if (totalSamplesEl) totalSamplesEl.innerText = metrics.total_samples;
    if (epochsEl) epochsEl.innerText = metrics.epochs_trained;

    // Atualiza tabela de relatório de classificação
    const tableBody = document.getElementById("classificationReportBody");
    if (tableBody && metrics.classification_report) {
      tableBody.innerHTML = "";
      const displayNames = metrics.display_names || [];

      displayNames.forEach(name => {
        const rowData = metrics.classification_report[name];
        if (rowData) {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>${name}</strong></td>
            <td>${(rowData.precision * 100).toFixed(1)}%</td>
            <td>${(rowData.recall * 100).toFixed(1)}%</td>
            <td>${(rowData['f1-score'] * 100).toFixed(1)}%</td>
            <td>${rowData.support}</td>
          `;
          tableBody.appendChild(tr);
        }
      });
    }

  } catch (e) {
    console.error("Erro ao carregar métricas:", e);
  }
}

/**
 * Inicializa botões manuais de teste de acessibilidade
 */
function initAccessibilityButtons() {
  const readBtn = document.getElementById("actionBtnRead");
  const fontBtn = document.getElementById("actionBtnFont");
  const decreaseFontBtn = document.getElementById("actionBtnDecreaseFont");
  const voiceBtn = document.getElementById("actionBtnVoice");
  const lightModeBtn = document.getElementById("actionBtnLightMode");
  const darkModeBtn = document.getElementById("actionBtnDarkMode");
  const contrastBtn = document.getElementById("actionBtnContrast");
  const contrastToggleBtn = document.getElementById("contrastToggleBtn");

  if (readBtn) readBtn.addEventListener("click", () => window.accessibilitySim.readCurrentText());
  if (fontBtn) fontBtn.addEventListener("click", () => window.accessibilitySim.increaseFontSize(0.15, true));
  if (decreaseFontBtn) decreaseFontBtn.addEventListener("click", () => window.accessibilitySim.decreaseFontSize(0.15, true));
  if (voiceBtn) voiceBtn.addEventListener("click", () => window.accessibilitySim.cycleVoice());
  if (lightModeBtn) lightModeBtn.addEventListener("click", () => window.accessibilitySim.setLightMode());
  if (darkModeBtn) darkModeBtn.addEventListener("click", () => window.accessibilitySim.setDarkMode());
  if (contrastBtn) contrastBtn.addEventListener("click", () => window.accessibilitySim.toggleTheme());
  if (contrastToggleBtn) contrastToggleBtn.addEventListener("click", () => window.accessibilitySim.toggleTheme());
}

/**
 * Inicializa controles do editor de texto personalizável da síntese vocal
 */
function initTextEditorControls() {
  const textarea = document.getElementById("simTextContent");
  const charCountEl = document.getElementById("simCharCount");
  const btnDefault = document.getElementById("btnPresetDefault");
  const btnClear = document.getElementById("btnClearText");
  const btnSpeak = document.getElementById("btnSpeakNow");
  const presetChips = document.querySelectorAll(".preset-chip");

  const defaultText = "O reconhecimento de voz baseado em coeficientes MFCC e redes neurais profundas proporciona autonomia digital sem precedentes para pessoas com deficiências motoras e visuais. Comandos simples como ler a tela, aumentar a tipografia ou ajustar o contraste transformam a web em um espaço verdadeiramente universal e acessível.";

  const presetTexts = {
    acessibilidade: "A acessibilidade digital garante que pessoas com deficiências visuais, auditivas ou motoras possam navegar, compreender e interagir com autonomia e igualdade na rede.",
    ia: "Redes neurais convolucionais combinadas com extração de coeficientes MFCC transformam ondas acústicas em representações matriciais e identificam comandos com alta precisão.",
    saudacao: "Olá! Seja bem-vindo ao sistema assistivo inteligente de reconhecimento de voz. Estou pronto para executar os seus comandos."
  };

  function updateCounts() {
    if (!textarea || !charCountEl) return;
    const val = textarea.value.trim();
    const words = val.length === 0 ? 0 : val.split(/\s+/).filter(Boolean).length;
    const chars = textarea.value.length;
    charCountEl.innerText = `${words} palavras • ${chars} caracteres`;
  }

  if (textarea) {
    textarea.addEventListener("input", updateCounts);
    updateCounts(); // Inicializa contagem de palavras e caracteres
  }

  // Botão Restaurar Padrão
  if (btnDefault) {
    btnDefault.addEventListener("click", () => {
      if (textarea) {
        textarea.value = defaultText;
        updateCounts();
        window.accessibilitySim.showToast("Texto padrão restaurado.", "cyan");
        textarea.focus();
      }
    });
  }

  // Botão Limpar
  if (btnClear) {
    btnClear.addEventListener("click", () => {
      if (textarea) {
        textarea.value = "";
        updateCounts();
        window.accessibilitySim.showToast("Campo de texto limpo. Digite sua frase personalizada.", "amber");
        textarea.focus();
      }
    });
  }

  // Botão Ouvir / Parar direto na barra do editor
  if (btnSpeak) {
    btnSpeak.addEventListener("click", () => {
      window.accessibilitySim.readCurrentText();
    });
  }

  // Chips de frases pré-definidas para teste ágil
  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const presetKey = chip.getAttribute("data-preset");
      if (presetTexts[presetKey] && textarea) {
        textarea.value = presetTexts[presetKey];
        updateCounts();
        window.accessibilitySim.showToast(`Exemplo carregado: ${chip.innerText}`, "cyan");
        textarea.focus();
      }
    });
  });
}

/**
 * Inicializa controles de zoom da interface (botões de passo rápido e slider)
 */
function initZoomControls() {
  const btnMinus = document.getElementById("btnZoomMinus");
  const btnPlus = document.getElementById("btnZoomPlus");
  const btnVoice = document.getElementById("btnZoomVoicePrompt");
  const slider = document.getElementById("fontScaleSlider");

  if (btnMinus) {
    btnMinus.addEventListener("click", () => {
      window.accessibilitySim.decreaseFontSize(0.15, true);
    });
  }

  if (btnPlus) {
    btnPlus.addEventListener("click", () => {
      window.accessibilitySim.increaseFontSize(0.15, true);
    });
  }

  if (btnVoice) {
    btnVoice.addEventListener("click", () => {
      window.accessibilitySim.promptVoiceFontSize('increase');
    });
  }

  if (slider) {
    slider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      window.accessibilitySim.setFontScale(val / 100, false);
    });

    slider.addEventListener("change", (e) => {
      const val = parseInt(e.target.value, 10);
      window.accessibilitySim.setFontScale(val / 100, true);
    });
  }
}

/**
 * Inicializa eventos do modal de ajuste de tamanho de fonte por voz
 */
function initVoiceFontModalEvents() {
  const modal = document.getElementById("voiceFontModal");
  const closeBtn = document.getElementById("btnCloseVoiceModal");
  const cancelBtn = document.getElementById("btnCancelVoiceModal");
  const applyBtn = document.getElementById("btnApplyVoiceFont");
  const restartBtn = document.getElementById("btnRestartVoiceListening");
  const manualInput = document.getElementById("voiceManualPercentInput");
  const computedText = document.getElementById("voiceComputedScaleText");
  const chips = document.querySelectorAll(".voice-chip");

  if (closeBtn) closeBtn.addEventListener("click", () => window.accessibilitySim.closeVoiceFontModal());
  if (cancelBtn) cancelBtn.addEventListener("click", () => window.accessibilitySim.closeVoiceFontModal());
  if (applyBtn) applyBtn.addEventListener("click", () => window.accessibilitySim.applyVoiceFontModal());
  if (restartBtn) restartBtn.addEventListener("click", () => window.accessibilitySim.startVoiceListening());

  // Input manual no modal
  if (manualInput) {
    manualInput.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10);
      if (!isNaN(val) && computedText) {
        computedText.innerText = `${val}%`;
        window.accessibilitySim.pendingVoiceScale = val;
      }
    });
  }

  // Chips de atalho rápido
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      let targetScale;
      const currentVal = manualInput ? parseInt(manualInput.value, 10) : Math.round(window.accessibilitySim.currentScale * 100);

      if (chip.hasAttribute("data-scale")) {
        targetScale = parseInt(chip.getAttribute("data-scale"), 10);
      } else if (chip.hasAttribute("data-delta")) {
        const delta = parseInt(chip.getAttribute("data-delta"), 10);
        targetScale = currentVal + delta;
      }

      if (targetScale) {
        targetScale = Math.min(220, Math.max(70, targetScale));
        if (manualInput) manualInput.value = targetScale;
        if (computedText) computedText.innerText = `${targetScale}%`;
        window.accessibilitySim.pendingVoiceScale = targetScale;
        window.accessibilitySim.showToast(`Valor definido para ${targetScale}%`, "cyan");
      }
    });
  });

  // Fechamento ao clicar fora do card
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        window.accessibilitySim.closeVoiceFontModal();
      }
    });
  }

  // Tecla Escape para fechar
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal && modal.style.display !== "none") {
      window.accessibilitySim.closeVoiceFontModal();
    }
  });
}
