# 🚦 Sistema de Semáforo Inteligente com YOLO11

Projeto de visão computacional desenvolvido para detecção em tempo real de veículos de emergência em ambientes de tráfego, utilizando o algoritmo **YOLO11s** treinado com imagens de maquete para controle priorizado de sinalização de trânsito.

---

## 📌 Visão Geral
O objetivo deste projeto é identificar veículos prioritários (como ambulâncias, viaturas e bombeiros) em vias urbanas para permitir a comutação inteligente e prioritária dos semáforos, reduzindo o tempo de resposta em emergências.

---

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Visão Computacional:** Ultralytics YOLO11 (`yolo11s.pt`) & OpenCV
* **Processamento:** PyTorch com suporte a GPU (CUDA)
* **Controle de Versão:** Git & GitHub

---

## 📁 Estrutura do Repositório

```text
├── dataset_maquete/       # Dataset de imagens e rótulos (ignorado pelo git)
├── runs/                   # Resultados dos treinamentos, gráficos e pesos (.pt)
│   └── detect/
│       └── treinamento_maquete/
│           └── modelo_emergencia/
│               └── weights/
│                   └── best.pt  # Pesos do melhor modelo treinado
├── exemplo.png             # Imagem de teste para predição
├── log_treinamento.txt     # Logs e histórico do treinamento
├── requirements.txt        # Dependências do projeto Python
├── treinar_yolo.py         # Script principal de treinamento da YOLO
└── README.md               # Documentação do projeto
