# 🚦 Sistema de Semáforo Inteligente com YOLO11

Projeto de visão computacional desenvolvido para detecção em tempo real de veículos de emergência e pedestres em ambientes de tráfego, utilizando o algoritmo **YOLO11s** para controle priorizado de sinalização de trânsito. 

O projeto é dividido em dois escopos principais:
* **Grupo 1:** Detecção e priorização de veículos de emergência.
* **Grupo 2:** Detecção e travessia segura de pedestres.

---

## 📌 Visão Geral

O objetivo geral deste projeto é desenvolver um protótipo educativo utilizando uma maquete para simulação de um cruzamento urbano com controle automático e inteligente de sinalização viária.

### Grupo 1: Veículos de Emergência
Identifica veículos prioritários (como ambulâncias, viaturas policiais e carros de bombeiros) para permitir a abertura prioritária dos semáforos, reduzindo o tempo de resposta em emergências. Para este grupo, foi realizado *fine-tuning* da arquitetura YOLO11 para focar exclusivamente na classe de emergência.

### Grupo 2: Pedestres
Identifica pedestres em faixas de pedestre localizadas nos cruzamentos para garantir uma travessia segura através da interrupção prioritária do tráfego veicular, focando na segurança do pedestre. O modelo padrão da YOLO já apresentou alta precisão na detecção de figuras humanas (mesmo na escala da maquete), necessitando apenas da integração dos sinais com o microcontrolador.

---

## ⚙️ Funcionamento e Integração

O protótipo é composto por uma maquete de cruzamento, miniaturas para simular veículos de emergência e bonecos para simular pedestres. Uma câmera fixa apontada para o cruzamento analisa quatro regiões predefinidas: **Norte (N), Sul (S), Leste (L) e Oeste (O)**. O sistema calcula a intersecção entre as *Bounding Boxes* (BB) das detecções e essas regiões delimitadas (ROI).

### Lógica de Controle:

* **Grupo 1 (Veículos):**
  * `VN` / `VS` (Veículo a Norte/Sul): Abre a via Norte-Sul e fecha a via Leste-Oeste.
  * `VL` / `VO` (Veículo a Leste/Oeste): Abre a via Leste-Oeste e fecha a via Norte-Sul.

* **Grupo 2 (Pedestres):**
  * `PN` / `PS` (Pedestre a Norte/Sul): Fecha a via Norte-Sul e abre a via Leste-Oeste.
  * `PL` / `PO` (Pedestre a Leste/Oeste): Fecha a via Leste-Oeste e abre a via Norte-Sul.

O controlador (Arduino) recebe esses comandos via comunicação serial e altera o estado dos semáforos em tempo real.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Visão Computacional:** Ultralytics YOLO11 (`yolo11s.pt`) & OpenCV
* **Processamento:** PyTorch com suporte a GPU (CUDA)
* **Hardware & Embarcados:** Microcontrolador Arduino & Comunicação Serial (`pyserial`)
* **Controle de Versão:** Git & GitHub

---

## 📁 Estrutura do Repositório

```text
├── runs/                   # Resultados dos treinamentos, gráficos e pesos (.pt)
│   └── detect/
│       └── treinamento_maquete/
│           └── modelo_emergencia/
│               └── weights/
│                   └── best.pt  # Pesos do melhor modelo treinado
├── main_grupo1_veiculos.py  # Script de execução principal (Grupo 1)
├── main_grupo2_pedestres.py # Script de execução principal (Grupo 2)
├── requirements.txt         # Dependências do projeto Python
├── treinar_yolo.py          # Script principal de treinamento da YOLO
└── README.md                # Documentação do projeto
```
## Utilização
Após fazer o clone do projeto você precisa se atentar a esses pontos:
Compatibilidade do arduino e verificação da porta de entrada.
Assim que rodar qualquer um dos projetos "main_grupo1_veiculos.py" para o de veiculos ou "main_grupo2_pessoas.py" para detecção de pedestres o programa te perguntará se deseja reanotar os pontos de Norte/Sul/Leste/Oeste na imagem. Cada vez que a câmera ou a maquete são movimentados isso precisa ser refeito. 

## Imagens do projeto
<img width="1280" height="960" alt="WhatsApp Image 2026-08-06 at 15 23 39" src="https://github.com/user-attachments/assets/f22ae8ea-8d34-4df9-8cd0-6a9ae57a5f4e" />

<img width="1280" height="960" alt="WhatsApp Image 2026-08-06 at 15 47 49" src="https://github.com/user-attachments/assets/dea6fa45-9048-4322-8a43-c94a4ad08729" />



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
   * **URL:** [Roboflow Universe](https://universe.roboflow.com/wasteclassification-tczus/emergency-vehicles-p6vwf/dataset/4)
   * **Licença:** CC BY 4.0
5. **emergency-vehicle**
   * **URL:** [Roboflow Universe](https://universe.roboflow.com/abdelouafi-boumoula/emergency-vehicle-wuhke)
   * **Licença:** CC BY 4.0

---


