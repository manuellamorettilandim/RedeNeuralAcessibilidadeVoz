"""
=============================================================================
PROJETO: Reconhecimento de Comandos de Voz para Acessibilidade
MÓDULO: model_arch.py - Arquitetura da Rede Neural Convolucional 2D (PyTorch)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceCommandCNN(nn.Module):
    """
    Rede Neural Convolucional 2D otimizada para reconhecimento de padrões
    espectro-temporais em matrizes MFCC (20 coeficientes x 47 janelas temporais).
    """

    def __init__(self, num_classes=6):
        super(VoiceCommandCNN, self).__init__()
        
        # Bloco Convolucional 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Bloco Convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bloco Convolucional 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        
        # Regularização e Camadas Densas (Fully Connected)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (Batch, 1, 20, 47)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits
