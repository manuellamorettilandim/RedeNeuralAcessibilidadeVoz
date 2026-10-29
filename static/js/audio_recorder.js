/**
 * =============================================================================
 * PROJETO: Reconhecimento de Comandos de Voz para Acessibilidade
 * MÓDULO: audio_recorder.js - Gravador de Microfone Web Audio API com Codificador WAV (16kHz)
 * =============================================================================
 */

class AudioRecorderService {
  constructor(targetSampleRate = 16000) {
    this.targetSampleRate = targetSampleRate;
    this.audioContext = null;
    this.mediaStream = null;
    this.analyser = null;
    this.processorNode = null;
    this.isRecording = false;
    this.audioBuffers = [];
    this.animationFrameId = null;
  }

  /**
   * Solicita permissão e inicializa o AudioContext
   */
  async init() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Seu navegador não suporta gravação de áudio via Web Audio API.");
    }
  }

  /**
   * Inicia a gravação com duração fixa e visualizador em tempo real
   * @param {number} durationSeconds - Duração em segundos (ex: 1.5s)
   * @param {HTMLCanvasElement} canvasElement - Canvas para visualização osciloscópio
   * @param {Function} onTick - Callback de contagem regressiva (tempo restante)
   * @param {Function} onComplete - Callback ao finalizar (retorna Blob WAV e array de amostras)
   * @param {Function} onError - Callback de erro
   */
  async startRecording(durationSeconds = 1.5, canvasElement = null, onTick = null, onComplete = null, onError = null) {
    if (this.isRecording) return;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false, // Preserva resposta acústica para MFCC
          autoGainControl: true
        }
      });

      window.AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContext();
      const inputSource = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Analisador para visualização osciloscópio
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 1024;
      inputSource.connect(this.analyser);

      // Buffer de gravação PCM
      this.audioBuffers = [];
      const bufferSize = 4096;
      this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

      this.processorNode.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        this.audioBuffers.push(new Float32Array(channelData));
      };

      inputSource.connect(this.processorNode);
      this.processorNode.connect(this.audioContext.destination);

      this.isRecording = true;

      // Inicia visualizador em Canvas
      if (canvasElement) {
        this.startCanvasVisualizer(canvasElement);
      }

      // Temporizador de gravação com contagem regressiva
      const startTime = Date.now();
      const totalMs = durationSeconds * 1000;

      const timerInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const remaining = Math.max(0, (totalMs - elapsed) / 1000);

        if (onTick) {
          onTick(remaining.toFixed(1));
        }

        if (elapsed >= totalMs) {
          clearInterval(timerInterval);
          this.stopRecording().then((wavBlob) => {
            if (onComplete) {
              onComplete(wavBlob);
            }
          }).catch(err => {
            if (onError) onError(err);
          });
        }
      }, 50);

    } catch (err) {
      this.cleanup();
      if (onError) onError(err);
    }
  }

  /**
   * Finaliza a captura de áudio e compila em WAV 16kHz
   */
  async stopRecording() {
    this.isRecording = false;

    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    // Concatena todos os buffers gravados
    const totalLength = this.audioBuffers.reduce((acc, b) => acc + b.length, 0);
    const mergedBuffer = new Float32Array(totalLength);
    let offset = 0;
    for (const buf of this.audioBuffers) {
      mergedBuffer.set(buf, offset);
      offset += buf.length;
    }

    const inputSampleRate = this.audioContext ? this.audioContext.sampleRate : 44100;

    // Resampleia para 16 kHz se necessário
    const resampledData = this.resampleBuffer(mergedBuffer, inputSampleRate, this.targetSampleRate);

    // Codifica para arquivo WAV PCM 16-bit mono
    const wavBlob = this.encodeWAV(resampledData, this.targetSampleRate);

    this.cleanup();
    return wavBlob;
  }

  /**
   * Libera microfone e conexões de áudio
   */
  cleanup() {
    this.isRecording = false;
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  /**
   * Resample simples e robusto por interpolação linear
   */
  resampleBuffer(audioData, fromRate, toRate) {
    if (fromRate === toRate) return audioData;
    const ratio = fromRate / toRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const originalPos = i * ratio;
      const leftIndex = Math.floor(originalPos);
      const rightIndex = Math.min(leftIndex + 1, audioData.length - 1);
      const fraction = originalPos - leftIndex;
      result[i] = audioData[leftIndex] * (1 - fraction) + audioData[rightIndex] * fraction;
    }
    return result;
  }

  /**
   * Cria o cabeçalho e corpo de um arquivo WAV 16-bit PCM mono
   */
  encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // Escreve string ASCII auxiliar
    const writeString = (view, offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    // RIFF identifier
    writeString(view, 0, 'RIFF');
    // RIFF chunk length
    view.setUint32(4, 36 + samples.length * 2, true);
    // RIFF type
    writeString(view, 8, 'WAVE');
    // format chunk identifier
    writeString(view, 12, 'fmt ');
    // format chunk length
    view.setUint32(16, 16, true);
    // sample format (1 = PCM)
    view.setUint16(20, 1, true);
    // channel count (1 = mono)
    view.setUint16(22, 1, true);
    // sample rate
    view.setUint32(24, sampleRate, true);
    // byte rate (sample rate * block align)
    view.setUint32(28, sampleRate * 2, true);
    // block align (channel count * bytes per sample)
    view.setUint16(32, 2, true);
    // bits per sample
    view.setUint16(34, 16, true);
    // data chunk identifier
    writeString(view, 36, 'data');
    // data chunk length
    view.setUint32(40, samples.length * 2, true);

    // Gravação das amostras PCM 16-bit
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  /**
   * Desenha a onda osciloscópica no Canvas em tempo real
   */
  startCanvasVisualizer(canvas) {
    const ctx = canvas.getContext('2d');
    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!this.isRecording) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }

      this.animationFrameId = requestAnimationFrame(draw);
      this.analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#38bdf8';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#38bdf8';

      ctx.beginPath();
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    draw();
  }
}

// Instância global
window.audioRecorder = new AudioRecorderService(16000);
