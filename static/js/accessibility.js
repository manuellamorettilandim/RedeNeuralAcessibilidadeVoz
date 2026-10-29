/**
 * =============================================================================
 * PROJETO: Reconhecimento de Comandos de Voz para Acessibilidade
 * MÓDULO: accessibility.js - Simulador Interativo de Acessibilidade
 * =============================================================================
 * Gerencia a execução das 4 ações principais de acessibilidade:
 * 1. "ler texto"       -> Leitor de tela por síntese de voz nativa (Web Speech API)
 * 2. "aumentar fonte"  -> Zoom tipográfico progressivo (+15% até 160% -> reset)
 * 3. "mudar voz"       -> Alterna entre vozes/timbres disponíveis no sistema
 * 4. "alto contraste"  -> Alterna o modo de alto contraste WCAG AAA
 * =============================================================================
 */

class AccessibilitySimulator {
  constructor() {
    this.fontScaleLevels = [1.0, 1.15, 1.30, 1.45, 1.60];
    this.currentScaleIndex = 0;
    this.currentScale = 1.0;
    this.isHighContrast = false;
    this.isLightMode = false;
    this.synth = window.speechSynthesis;
    this.voices = [];
    this.currentVoiceIndex = 0;
    this.isSpeaking = false;
    this.speechPitches = [1.0, 1.3, 0.8, 1.1];
    this.currentPitchIndex = 0;

    // Reconhecimento de fala para quantidades
    this.recognition = null;
    this.isListeningVoice = false;
    this.voicePromptDirection = 'increase';
    this.pendingVoiceScale = 100;

    this.initVoices();
    this.initSpeechRecognition();
  }

  /**
   * Carrega vozes nativas da Web Speech API
   */
  initVoices() {
    if (!this.synth) return;

    const populate = () => {
      this.voices = this.synth.getVoices().filter(v => v.lang.includes('pt') || v.lang.includes('en') || v.lang.includes('es'));
      if (this.voices.length === 0) {
        this.voices = this.synth.getVoices();
      }
    };

    populate();
    if (this.synth.onvoiceschanged !== undefined) {
      this.synth.onvoiceschanged = populate;
    }
  }

  /**
   * Executa a ação mapeada a partir da predição da rede neural ou interação
   */
  executeAction(actionType) {
    switch (actionType) {
      case 'read_text':
        this.readCurrentText();
        break;
      case 'increase_font':
        this.increaseFontSize(0.15, true);
        break;
      case 'decrease_font':
        this.decreaseFontSize(0.15, true);
        break;
      case 'change_voice':
        this.cycleVoice();
        break;
      case 'set_light_theme':
      case 'modo_claro':
        this.setLightMode();
        break;
      case 'set_dark_theme':
      case 'modo_escuro':
        this.setDarkMode();
        break;
      case 'high_contrast':
      case 'toggle_theme':
        this.toggleTheme();
        break;
      default:
        console.log("Nenhuma ação associada para:", actionType);
        break;
    }
  }

  /**
   * Interrompe qualquer síntese de voz em andamento
   */
  stopSpeaking() {
    if (this.synth) {
      this.synth.cancel();
    }
    this.isSpeaking = false;
    this.updateSpeakingUI(false);
    this.showToast("Leitura de voz interrompida.", "cyan");
  }

  /**
   * Atualiza botões e destaques visuais do estado de leitura
   */
  updateSpeakingUI(isSpeaking) {
    const textElement = document.getElementById("simTextContent");
    if (textElement) {
      if (isSpeaking) {
        textElement.classList.add("reading-highlight");
      } else {
        textElement.classList.remove("reading-highlight");
      }
    }

    const speakBtn = document.getElementById("btnSpeakNow");
    const speakBtnIcon = document.getElementById("speakBtnIcon");
    const speakBtnText = document.getElementById("speakBtnText");
    const actionBtnRead = document.getElementById("actionBtnRead");

    if (speakBtn) {
      speakBtn.classList.toggle("speaking", isSpeaking);
    }
    if (speakBtnIcon) {
      speakBtnIcon.innerText = isSpeaking ? "⏹️" : "🔊";
    }
    if (speakBtnText) {
      speakBtnText.innerText = isSpeaking ? "Parar" : "Ouvir";
    }

    if (actionBtnRead) {
      actionBtnRead.classList.toggle("active-reading", isSpeaking);
      actionBtnRead.innerHTML = isSpeaking
        ? `<span>⏹️</span> "parar leitura"`
        : `<span>🔊</span> "ler texto" (TTS)`;
    }
  }

  /**
   * AÇÃO 1: Ler Texto em Voz Alta (Web Speech API) com destaque de leitura
   * Lê dinamicamente o texto digitado ou colado pelo usuário no campo editável
   */
  readCurrentText() {
    if (!this.synth) {
      this.showToast("Síntese de voz não suportada pelo navegador.", "amber");
      return;
    }

    // Se já estiver falando, o acionamento funciona como toggle (pausa/cancela)
    if (this.isSpeaking || this.synth.speaking) {
      this.stopSpeaking();
      return;
    }

    const textElement = document.getElementById("simTextContent");
    const rawText = textElement
      ? (textElement.value !== undefined ? textElement.value : textElement.innerText)
      : "";
    const textToRead = rawText.trim();

    if (!textToRead) {
      this.showToast("⚠️ Digite um texto no campo antes de iniciar a leitura.", "amber");
      if (textElement && typeof textElement.focus === "function") {
        textElement.focus();
      }
      return;
    }

    const utterance = new SpeechSynthesisUtterance(textToRead);
    utterance.lang = "pt-BR";
    utterance.rate = 1.0;
    utterance.pitch = this.speechPitches[this.currentPitchIndex];

    if (this.voices.length > 0) {
      utterance.voice = this.voices[this.currentVoiceIndex % this.voices.length];
    }

    this.playTone(523.25, 0.1); // Tom C5 de confirmação
    this.isSpeaking = true;
    this.updateSpeakingUI(true);

    utterance.onend = () => {
      this.isSpeaking = false;
      this.updateSpeakingUI(false);
      this.showToast("Leitura concluída.", "cyan");
    };

    utterance.onerror = (e) => {
      console.warn("Erro na síntese vocal:", e);
      this.isSpeaking = false;
      this.updateSpeakingUI(false);
    };

    this.synth.speak(utterance);
    this.showToast("Lendo texto em voz alta...", "emerald");
  }

  /**
   * Define a escala tipográfica de forma contínua e precisa (entre 70% e 220%)
   * @param {number} scale - Multiplicador (ex: 1.0 para 100%, 1.25 para 125%)
   * @param {boolean} announce - Se deve sintetizar fala de confirmação
   */
  setFontScale(scale, announce = false) {
    const clamped = Math.min(2.20, Math.max(0.70, Math.round(scale * 100) / 100));
    this.currentScale = clamped;
    const percentInt = Math.round(clamped * 100);

    document.documentElement.style.setProperty('--font-scale-factor', clamped);

    // Atualiza badges em toda a interface
    const scaleBadges = document.querySelectorAll("#fontScaleBadge, #simFontScaleBadge, .font-scale-display");
    scaleBadges.forEach(badge => {
      badge.innerText = `${percentInt}%`;
    });

    // Atualiza slider visual
    const slider = document.getElementById("fontScaleSlider");
    if (slider) {
      slider.value = percentInt;
    }

    // Atualiza valores no modal de voz se estiver aberto
    const modalComputedText = document.getElementById("voiceComputedScaleText");
    const modalInput = document.getElementById("voiceManualPercentInput");
    if (modalComputedText) modalComputedText.innerText = `${percentInt}%`;
    if (modalInput) modalInput.value = percentInt;

    // Feedback sonoro (frequência varia com o zoom)
    const toneFreq = 400 + (clamped - 0.7) * 300;
    this.playTone(toneFreq, 0.1);

    this.showToast(`Tamanho da fonte ajustado para ${percentInt}%`, "emerald");

    if (announce && this.synth) {
      this.synth.cancel();
      const msg = new SpeechSynthesisUtterance(`Fonte ajustada para ${percentInt} por cento.`);
      msg.lang = "pt-BR";
      msg.pitch = this.speechPitches[this.currentPitchIndex];
      if (this.voices[this.currentVoiceIndex]) msg.voice = this.voices[this.currentVoiceIndex];
      this.synth.speak(msg);
    }
  }

  /**
   * AÇÃO: Aumentar Tamanho da Fonte
   */
  increaseFontSize(amount = 0.15, announce = false) {
    this.setFontScale(this.currentScale + amount, announce);
  }

  /**
   * AÇÃO: Diminuir Tamanho da Fonte
   */
  decreaseFontSize(amount = 0.15, announce = false) {
    this.setFontScale(this.currentScale - amount, announce);
  }

  /**
   * AÇÃO 2: Aumentar Tamanho da Fonte Progressivamente (Compatibilidade)
   */
  cycleFontSize() {
    this.increaseFontSize(0.15, false);
  }

  /**
   * Abre o assistente de voz para o usuário falar a quantidade desejada de aumento/diminuição
   * @param {'increase'|'decrease'} direction
   */
  promptVoiceFontSize(direction = 'increase') {
    this.voicePromptDirection = direction;
    const modal = document.getElementById("voiceFontModal");
    if (!modal) {
      // Fallback: se o modal não existir, ajusta diretamente
      if (direction === 'decrease') this.decreaseFontSize(0.15, true);
      else this.increaseFontSize(0.15, true);
      return;
    }

    modal.style.display = "flex";

    const titleEl = document.getElementById("voiceFontModalTitle");
    const subEl = document.getElementById("voiceModalSubtitle");
    const transcriptText = document.getElementById("voiceTranscriptText");
    const computedText = document.getElementById("voiceComputedScaleText");
    const inputEl = document.getElementById("voiceManualPercentInput");

    const currentPercent = Math.round(this.currentScale * 100);
    const initialTarget = direction === 'decrease'
      ? Math.max(70, currentPercent - 15)
      : Math.min(220, currentPercent + 15);

    this.pendingVoiceScale = initialTarget;

    if (titleEl) {
      titleEl.innerText = direction === 'decrease'
        ? "Diminuir Tamanho da Fonte por Voz"
        : "Aumentar Tamanho da Fonte por Voz";
    }

    if (subEl) {
      subEl.innerHTML = direction === 'decrease'
        ? `Fale quanto deseja diminuir (ex: <em>"15%"</em>, <em>"diminuir 20%"</em>, <em>"85%"</em>)`
        : `Fale quanto deseja aumentar (ex: <em>"20%"</em>, <em>"aumentar 30%"</em>, <em>"130%"</em>)`;
    }

    if (transcriptText) {
      transcriptText.innerText = "Aguardando você falar a porcentagem...";
      transcriptText.classList.add("transcript-placeholder");
    }

    if (computedText) computedText.innerText = `${initialTarget}%`;
    if (inputEl) inputEl.value = initialTarget;

    // Pergunta por voz com sintetizador
    if (this.synth) {
      this.synth.cancel();
      const promptText = direction === 'decrease'
        ? "Quanto deseja diminuir a fonte? Fale a quantidade, por exemplo: 15 por cento."
        : "Quanto deseja aumentar a fonte? Fale a quantidade, por exemplo: 20 por cento.";

      const promptUtterance = new SpeechSynthesisUtterance(promptText);
      promptUtterance.lang = "pt-BR";
      promptUtterance.pitch = this.speechPitches[this.currentPitchIndex];
      if (this.voices[this.currentVoiceIndex]) promptUtterance.voice = this.voices[this.currentVoiceIndex];

      promptUtterance.onend = () => {
        this.startVoiceListening();
      };
      promptUtterance.onerror = () => {
        this.startVoiceListening();
      };

      this.synth.speak(promptUtterance);
    } else {
      this.startVoiceListening();
    }
  }

  /**
   * Inicializa o reconhecimento de fala para escuta da quantidade falada
   */
  initSpeechRecognition() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      console.log("Web Speech Recognition não suportado neste navegador. Controles manuais disponíveis.");
      return;
    }

    try {
      this.recognition = new SpeechRec();
      this.recognition.lang = "pt-BR";
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.maxAlternatives = 3;

      this.recognition.onstart = () => {
        this.isListeningVoice = true;
        const statusEl = document.getElementById("voiceStatusBadge");
        const pulseEl = document.getElementById("voicePulseIndicator");
        if (statusEl) statusEl.innerText = "Ouvindo seu microfone... Fale a porcentagem agora!";
        if (pulseEl) pulseEl.classList.add("active");
      };

      this.recognition.onresult = (event) => {
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          finalTranscript += event.results[i][0].transcript;
        }

        const transcriptEl = document.getElementById("voiceTranscriptText");
        if (transcriptEl && finalTranscript.trim()) {
          transcriptEl.innerText = `"${finalTranscript.trim()}"`;
          transcriptEl.classList.remove("transcript-placeholder");
        }

        const parsed = this.parseSpokenPercentage(finalTranscript, this.voicePromptDirection);
        if (parsed !== null) {
          const computedEl = document.getElementById("voiceComputedScaleText");
          const inputEl = document.getElementById("voiceManualPercentInput");
          if (computedEl) computedEl.innerText = `${parsed}%`;
          if (inputEl) inputEl.value = parsed;
          this.pendingVoiceScale = parsed;
        }
      };

      this.recognition.onerror = (event) => {
        this.isListeningVoice = false;
        const statusEl = document.getElementById("voiceStatusBadge");
        const pulseEl = document.getElementById("voicePulseIndicator");
        if (statusEl) statusEl.innerText = "Microfone em pausa. Escolha um atalho ou clique em 'Falar Novamente'.";
        if (pulseEl) pulseEl.classList.remove("active");
      };

      this.recognition.onend = () => {
        this.isListeningVoice = false;
        const pulseEl = document.getElementById("voicePulseIndicator");
        const statusEl = document.getElementById("voiceStatusBadge");
        if (pulseEl) pulseEl.classList.remove("active");
        if (statusEl) statusEl.innerText = "Fala processada! Clique em 'Aplicar Tamanho' ou confirme.";
      };
    } catch (e) {
      console.warn("Falha ao configurar SpeechRecognition:", e);
    }
  }

  /**
   * Inicia a escuta por microfone no modal de voz
   */
  startVoiceListening() {
    if (!this.recognition) return;
    if (this.isListeningVoice) {
      try { this.recognition.stop(); } catch (e) {}
    }
    try {
      this.recognition.start();
    } catch (e) {
      // Ignora se já estiver ativo
    }
  }

  /**
   * Para a escuta do microfone
   */
  stopVoiceListening() {
    if (this.recognition && this.isListeningVoice) {
      try {
        this.recognition.stop();
      } catch (e) {}
    }
    this.isListeningVoice = false;
  }

  /**
   * Parser inteligente de números e porcentagens em Português
   */
  parseSpokenPercentage(rawText, direction = 'increase') {
    if (!rawText) return null;
    const text = rawText.toLowerCase().trim();

    const numberWords = {
      "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9,
      "dez": 10, "onze": 11, "doze": 12, "treze": 13, "quatorze": 14, "catorze": 14, "quinze": 15,
      "dezesseis": 16, "dezessete": 17, "dezoito": 18, "dezenove": 19,
      "vinte": 20, "vinte e cinco": 25, "trinta": 30, "trinta e cinco": 35,
      "quarenta": 40, "quarenta e cinco": 45, "cinquenta": 50,
      "sessenta": 60, "setenta": 70, "oitenta": 80, "noventa": 90,
      "cem": 100, "cento e dez": 110, "cento e vinte": 120, "cento e trinta": 130,
      "cento e quarenta": 140, "cento e cinquenta": 150, "cento e sessenta": 160,
      "cento e setenta": 170, "cento e oitenta": 180, "cento e noventa": 190,
      "duzentos": 200
    };

    let detectedNumber = null;

    // Busca dígitos numéricos explícitos (ex: "20%", "30", "150%")
    const digitMatch = text.match(/(\d{1,3})\s*(%|por cento)?/);
    if (digitMatch) {
      detectedNumber = parseInt(digitMatch[1], 10);
    } else {
      // Busca números por extenso
      const sortedKeys = Object.keys(numberWords).sort((a, b) => b.length - a.length);
      for (const word of sortedKeys) {
        if (text.includes(word)) {
          detectedNumber = numberWords[word];
          break;
        }
      }
    }

    if (detectedNumber === null || isNaN(detectedNumber)) {
      return null;
    }

    const currentPercent = Math.round(this.currentScale * 100);

    const isExplicitIncrease = text.includes("aumentar") || text.includes("mais") || text.includes("elevar") || text.includes("subir");
    const isExplicitDecrease = text.includes("diminuir") || text.includes("menos") || text.includes("reduzir") || text.includes("baixar");

    // Se o usuário falou um valor absoluto alto (ex: 70 a 220) sem falar "mais" ou "menos", assume valor final
    if (detectedNumber >= 70 && !isExplicitIncrease && !isExplicitDecrease) {
      return Math.min(220, Math.max(70, detectedNumber));
    }

    // Caso contrário, calcula relativo
    if (isExplicitDecrease || (!isExplicitIncrease && direction === 'decrease')) {
      const result = currentPercent - detectedNumber;
      return Math.min(220, Math.max(70, result));
    } else {
      const result = currentPercent + detectedNumber;
      return Math.min(220, Math.max(70, result));
    }
  }

  /**
   * Fecha o modal de ajuste de voz
   */
  closeVoiceFontModal() {
    this.stopVoiceListening();
    const modal = document.getElementById("voiceFontModal");
    if (modal) {
      modal.style.display = "none";
    }
  }

  /**
   * Aplica a escala definida no modal de voz
   */
  applyVoiceFontModal() {
    const inputEl = document.getElementById("voiceManualPercentInput");
    const targetVal = inputEl ? parseInt(inputEl.value, 10) : this.pendingVoiceScale;
    if (!isNaN(targetVal) && targetVal >= 70 && targetVal <= 220) {
      this.setFontScale(targetVal / 100, true);
    }
    this.closeVoiceFontModal();
  }

  /**
   * AÇÃO 3: Mudar Perfil e Timbre Vocal
   */
  cycleVoice() {
    this.currentVoiceIndex = (this.currentVoiceIndex + 1) % Math.max(1, this.voices.length);
    this.currentPitchIndex = (this.currentPitchIndex + 1) % this.speechPitches.length;

    const selectedVoice = this.voices[this.currentVoiceIndex] ? this.voices[this.currentVoiceIndex].name : "Voz Padrão";
    const voiceBadge = document.getElementById("voiceProfileBadge");
    if (voiceBadge) {
      voiceBadge.innerText = `Perfil ${this.currentVoiceIndex + 1}: ${selectedVoice.substring(0, 15)}...`;
    }

    this.playTone(783.99, 0.15); // Tom G5

    // Fala uma mensagem curta com a nova voz para confirmação
    if (this.synth) {
      this.synth.cancel();
      const testUtterance = new SpeechSynthesisUtterance("Perfil de voz atualizado.");
      testUtterance.lang = "pt-BR";
      testUtterance.pitch = this.speechPitches[this.currentPitchIndex];
      if (this.voices[this.currentVoiceIndex]) {
        testUtterance.voice = this.voices[this.currentVoiceIndex];
      }
      this.synth.speak(testUtterance);
    }

    this.showToast(`Voz alterada: Perfil ${this.currentVoiceIndex + 1}`, "purple");
  }

  /**
   * Atualiza a interface gráfica, botões e badges para o tema atual
   */
  updateThemeUI() {
    document.body.classList.toggle("light-theme", this.isLightMode);

    const themeBtn = document.getElementById("contrastToggleBtn");
    if (themeBtn) {
      themeBtn.innerHTML = this.isLightMode
        ? `<span>🌙</span> Modo Escuro`
        : `<span>☀️</span> Modo Claro`;
      themeBtn.title = this.isLightMode ? "Alternar para Modo Escuro" : "Alternar para Modo Claro";
    }

    const themeBadge = document.getElementById("contrastModeBadge");
    if (themeBadge) {
      themeBadge.innerText = this.isLightMode ? "Modo Claro" : "Modo Escuro";
      themeBadge.style.color = this.isLightMode ? "var(--accent-cyan)" : "var(--text-muted)";
    }

    const actionThemeBtn = document.getElementById("actionBtnContrast");
    if (actionThemeBtn) {
      actionThemeBtn.innerHTML = this.isLightMode
        ? `<span>🌙</span> "modo escuro"`
        : `<span>☀️</span> "modo claro"`;
    }
  }

  /**
   * AÇÃO: Ativar Modo Claro
   */
  setLightMode(announce = true) {
    if (this.isLightMode) {
      this.showToast("O Modo Claro já está ativo.", "amber");
      return;
    }
    this.isLightMode = true;
    this.updateThemeUI();

    this.playTone(880.00, 0.15); // Tom A5
    this.showToast("Modo Claro Ativado", "amber");

    if (announce && this.synth) {
      this.synth.cancel();
      const testUtterance = new SpeechSynthesisUtterance("Modo claro ativado.");
      testUtterance.lang = "pt-BR";
      testUtterance.pitch = this.speechPitches[this.currentPitchIndex];
      if (this.voices[this.currentVoiceIndex]) {
        testUtterance.voice = this.voices[this.currentVoiceIndex];
      }
      this.synth.speak(testUtterance);
    }
  }

  /**
   * AÇÃO: Ativar Modo Escuro
   */
  setDarkMode(announce = true) {
    if (!this.isLightMode) {
      this.showToast("O Modo Escuro já está ativo.", "cyan");
      return;
    }
    this.isLightMode = false;
    this.updateThemeUI();

    this.playTone(659.25, 0.15); // Tom E5
    this.showToast("Modo Escuro Ativado", "cyan");

    if (announce && this.synth) {
      this.synth.cancel();
      const testUtterance = new SpeechSynthesisUtterance("Modo escuro ativado.");
      testUtterance.lang = "pt-BR";
      testUtterance.pitch = this.speechPitches[this.currentPitchIndex];
      if (this.voices[this.currentVoiceIndex]) {
        testUtterance.voice = this.voices[this.currentVoiceIndex];
      }
      this.synth.speak(testUtterance);
    }
  }

  /**
   * AÇÃO 4: Alternar entre Modo Claro e Modo Escuro
   */
  toggleTheme() {
    if (this.isLightMode) {
      this.setDarkMode();
    } else {
      this.setLightMode();
    }
  }

  /**
   * Alias para manter compatibilidade com a rede neural e chamadas legadas
   */
  toggleHighContrast() {
    this.toggleTheme();
  }

  /**
   * Reproduz um bipe sonoro suave de feedback auditivo
   */
  playTone(freq = 440, duration = 0.1) {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, ctx.currentTime);

      gain.gain.setValueAtTime(0.08, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + duration);
    } catch (e) {
      // Ignora silenciosamente se o áudio não estiver liberado
    }
  }

  /**
   * Exibe notificação flutuante acessível (Toast)
   */
  showToast(message, type = "cyan") {
    let container = document.getElementById("toastContainer");
    if (!container) {
      container = document.createElement("div");
      container.id = "toastContainer";
      container.className = "toast-container";
      document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.setAttribute("role", "status");
    toast.innerHTML = `<span>✦</span><span>${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(100%)";
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }
}

// Instância global
window.accessibilitySim = new AccessibilitySimulator();
