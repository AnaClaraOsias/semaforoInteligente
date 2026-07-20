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

## 📦 Datasets e Créditos

O dataset final utilzado para o treinamento do modelo foi reunido, filtrado e re-rotulado a partir de fontes públicas do **Roboflow Universe**, unificando todas as anotações em uma **única classe alvo (`Emergencia`)** no formato YOLOv11.

Agradecemos aos criadores e mantenedores dos datasets originais:

1. **Ambulance Police Firetruck**
   * **URL:** [Roboflow Universe](https://universe.roboflow.com/detection-cars/ambulance-police-firetruck)
   * **Licença:** CC BY 4.0
2. **Emergency Vehicle**
   * **URL:** [Roboflow Universe](https://universe.roboflow.com/ai-powered-traffic-management-system/emergency-vehicle-psv0q)
   * **Licença:** CC BY 4.0
3. **Emergency Vehicles - v4**
   * **Volume:** 12.865 imagens em formato YOLOv11
   * **Pré-processamento:** Auto-orientação e dimensionamento para 640x640 (com bordas pretas).
   * **Aumentações (Augmentations):** Espelhamento horizontal (50%), recorte aleatório (0-10%), rotação (-10° a +10°), cisalhamento (shear), ajuste de brilho/exposição, *Gaussian blur* e ruído *Salt and pepper*.
4. **emergency-vehicle**
   * **URL:** [Roboflow Universe](https://universe.roboflow.com/abdelouafi-boumoula/emergency-vehicle-wuhke)
   * **Licença:** CC BY 4.0

---
