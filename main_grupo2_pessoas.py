import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import json
import time
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO
import supervision as sv

# ==========================================
# 1. CONFIGURAÇÕES E CARREGAMENTO DO MODELO
# ==========================================
MODEL_PATH = "yolo11s.pt"  # Modelo YOLO padrão (pré-treinado no COCO)
CONFIDENCE_THRESHOLD = 0.35
TARGET_CLASS_ID = 0        # 0 é o ID para 'person' no dataset COCO
TARGET_CLASS_NAME = "Pessoa"
REGIOES_FILE = "faixas_pedestres.json"

# CONFIGURAÇÕES DE DESEMPENHO E VOTAÇÃO
FRAME_STRIDE = 2         # Inferência executada a cada 2 frames (conforme seu código original)
VOTING_WINDOW_SIZE = 20  # Tamanho da memória de quadros para estabilizar a decisão
VOTING_THRESHOLD = 0.50  # Exige 50% de dominância de presença para fechar o sinal

print("Carregando modelo YOLO (Inferência Direta - Foco em Pessoas)...")
model = YOLO(MODEL_PATH)

import serial

# ==========================================
# CONEXÃO SERIAL REAL COM O ARDUINO
# ==========================================
# Altere 'COM3' para a porta onde o Arduino está conectado no seu PC 
# (Exemplo no Windows: 'COM3', 'COM4' | no Linux/Mac: '/dev/ttyUSB0', '/dev/ttyACM0')
PORTA_SERIAL = 'COM3'
BAUD_RATE = 9600

try:
    print(f"Conectando ao Arduino na porta {PORTA_SERIAL}...")
    arduino = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=1)
    time.sleep(2)  # Tempo necessário para o Arduino reiniciar após abrir a porta serial
    print("✅ Conexão Serial estabelecida com sucesso!")
except Exception as e:
    print(f"⚠️ Não foi possível abrir a porta serial {PORTA_SERIAL}: {e}")
    arduino = None

ultimo_comando_enviado = ""

def enviar_comando_arduino(comando):
    global ultimo_comando_enviado
    if comando != ultimo_comando_enviado:
        print(f"--> [SERIAL REAL] Enviando comando: {comando}")
        
        # Envia via serial se a conexão estiver aberta
        if arduino is not None and arduino.is_open:
            # Converte a string (ex: "PN\n") para bytes antes de enviar
            arduino.write(f"{comando}\n".encode('utf-8'))
            
        ultimo_comando_enviado = comando

# ==========================================
# 3. MARCAÇÃO E MANIPULAÇÃO DE REGIÕES
# ==========================================
regions = {"norte": [], "sul": [], "leste": [], "oeste": []}
order = ["norte", "sul", "leste", "oeste"]
temp_points = []

def salvar_regioes(caminho, regioes):
    with open(caminho, "w") as f:
        json.dump(regioes, f, indent=4)
    print(f"✅ Regiões (faixas) salvas com sucesso em '{caminho}'.")

def carregar_regioes(caminho):
    with open(caminho, "r") as f:
        regioes = json.load(f)
    print(f"📂 Regiões carregadas de '{caminho}'.")
    return regioes

def mouse_callback(event, x, y, flags, param):
    global temp_points
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_points.append((x, y))

def annotate_regions_local(cap):
    global temp_points, regions
    current_region_idx = 0
    temp_points = []
    regions = {"norte": [], "sul": [], "leste": [], "oeste": []}

    ret, frame = cap.read()
    if not ret: raise RuntimeError("Não foi possível acessar a webcam.")
        
    window_name = "Marcacao de Faixas de Pedestre (Clique nos pontos)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    colors = {"norte": (0, 0, 255), "sul": (255, 0, 0), "leste": (0, 255, 0), "oeste": (0, 165, 255)}

    while current_region_idx < len(order):
        reg_name = order[current_region_idx]
        canvas = frame.copy()

        for r_name, pts in regions.items():
            if len(pts) > 0:
                cv2.polylines(canvas, [np.array(pts, np.int32)], True, colors[r_name], 2)

        if len(temp_points) > 0:
            for pt in temp_points:
                cv2.circle(canvas, pt, 4, colors[reg_name], -1)
            if len(temp_points) > 1:
                cv2.polylines(canvas, [np.array(temp_points, np.int32)], False, colors[reg_name], 2)

        cv2.putText(canvas, f"Desenhe a faixa: {reg_name.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Pressione 'ENTER' para confirmar | 'C' para limpar", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 10] and len(temp_points) >= 3:
            regions[reg_name] = temp_points.copy()
            temp_points = []
            current_region_idx += 1
        elif key == ord('c'):
            temp_points = []

    cv2.destroyWindow(window_name)

# ==========================================
# 4. DECISÃO DE REGIÃO
# ==========================================
# ==========================================
# 4. DECISÃO DE REGIÃO PELA BASE (PÉS)
# ==========================================
def bb_best_region(box, regions):
    x1, y1, x2, y2 = map(int, box)
    
    largura = max(1, x2 - x1)
    best_region = None
    max_frac = 0.0

    for nome, pts in regions.items():
        if len(pts) < 3: continue
        pts_np = np.array(pts, dtype=np.int32)
        
        pontos_dentro = 0
        num_samples = 10 # Testa 10 pontos ao longo da base do BB
        
        for i in range(num_samples + 1):
            px = x1 + int((largura * i) / num_samples)
            py = y2
            # Verifica se a coordenada (px, py) do pé está dentro do polígono
            if cv2.pointPolygonTest(pts_np, (px, py), False) >= 0:
                pontos_dentro += 1
        
        # Calcula a porcentagem do pé que está dentro da faixa
        frac = pontos_dentro / (num_samples + 1)
        
        if frac > max_frac:
            max_frac = frac
            best_region = nome

    # Retorna a região se pelo menos uma parte do pé (ex: > 0%) estiver nela
    return best_region if max_frac > 0 else None


# ==========================================
# 5. LÓGICA DE INFERÊNCIA DIRETA
# ==========================================
def executar_inferencia(frame):
    # Inferência com filtro apenas para classe 0 (pessoa)
    results = model.predict(source=frame, classes=[TARGET_CLASS_ID], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filtro extra de confiança
    if len(detections) > 0:
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        
    return detections

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# ==========================================
# 6. LOOP PRINCIPAL
# ==========================================
def main():
    global regions

    if os.path.exists(REGIOES_FILE):
        resposta = input(f"\nJá existe um arquivo '{REGIOES_FILE}'. Deseja reanotar as faixas? [s/N]: ").strip().lower()
        if resposta == 's':
            cap_temp = cv2.VideoCapture(0)
            if not cap_temp.isOpened(): return
            annotate_regions_local(cap_temp)
            salvar_regioes(REGIOES_FILE, regions)
            cap_temp.release()
        else:
            regions = carregar_regioes(REGIOES_FILE)
    else:
        cap_temp = cv2.VideoCapture(0)
        if not cap_temp.isOpened(): return
        annotate_regions_local(cap_temp)
        salvar_regioes(REGIOES_FILE, regions)
        cap_temp.release()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    print("Aguardando estabilização da câmera...")
    time.sleep(1.5) 
    for _ in range(5): cap.read()

    # Mapa de comandos: 'F' para Fechar sinal, 'ABERTO' indica sinal liberado para carros
    mapa_comandos = {'norte': 'FN', 'sul': 'FS', 'leste': 'FL', 'oeste': 'FO'}

    janela_nome = "Sistema - Detecao de Pedestres"
    cv2.namedWindow(janela_nome)
    cv2.waitKey(1)

    frame_count = 0
    detections = sv.Detections.empty()
    labels = []
    
    # JANELA DE MEMÓRIA DE VOTOS (Para evitar oscilação)
    historico_votos = deque(maxlen=VOTING_WINDOW_SIZE)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            curr_time = time.time()
            elapsed_time = curr_time - prev_time
            prev_time = curr_time
            if elapsed_time > 0: fps = 1.0 / elapsed_time

            # Pula frames de acordo com o FRAME_STRIDE
            if frame_count % FRAME_STRIDE != 0 and len(detections) > 0:
                pass 
            else:
                detections = executar_inferencia(frame)
                labels = []
                regioes_frame_atual = []

                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                    regiao = bb_best_region((x1, y1, x2, y2), regions)

                    if regiao:
                        regioes_frame_atual.append(regiao)
                        labels.append(f"{regiao.upper()}")
                    else:
                        labels.append(f"Fora da faixa")

                # Salva o voto deste frame (Se há pessoa em alguma região, pega a primeira detectada)
                voto_atual = regioes_frame_atual[0] if len(regioes_frame_atual) > 0 else "NENHUM"
                historico_votos.append(voto_atual)

            # LÓGICA: Se tem pessoa na faixa de forma consistente, fecha o sinal
            if len(historico_votos) > 0:
                contagem = Counter(historico_votos)
                mais_voted_regiao, qtd_votos = contagem.most_common(1)[0]
                porcentagem_votos = qtd_votos / len(historico_votos)

                if mais_voted_regiao != "NENHUM" and porcentagem_votos >= VOTING_THRESHOLD:
                    regiao_confirmada = mais_voted_regiao
                    comando_decidido = mapa_comandos[regiao_confirmada]
                else:
                    regiao_confirmada = None
                    comando_decidido = "ABERTO" # Sinal aberto para os carros
            else:
                regiao_confirmada = None
                comando_decidido = "ABERTO"

            # Envia comando para a simulação serial
            enviar_comando_arduino(comando_decidido)

            # ==========================================
            # DESENHO DA INTERFACE
            # ==========================================
            annotated_frame = frame.copy()
            
            # Desenha as Bounding Boxes completas e os Rótulos
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            # ---> NOVO: Desenha a linha vermelha destacando a base (os pés) <---
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                cv2.line(annotated_frame, (x1, y2), (x2, y2), (0, 0, 255), 3) # Linha vermelha grossa na base

            fps_str = f"FPS: {fps:.0f}"
            
            # Atualiza HUD dependendo se o sinal deve fechar ou não
            if regiao_confirmada:
                status_txt = f"SINAL FECHADO: PEDESTRE EM {regiao_confirmada.upper()} ({comando_decidido}) | {fps_str}"
                cor_indicador = (0, 0, 255)  # Vermelho (sinal fechado para carros)
            else:
                status_txt = f"SINAL ABERTO ({comando_decidido}) | {fps_str}"
                cor_indicador = (0, 255, 0)  # Verde (sinal liberado para carros)

            cv2.rectangle(annotated_frame, (10, 10), (450, 40), (20, 20, 20), -1)
            cv2.circle(annotated_frame, (25, 25), 6, cor_indicador, -1)
            
            cv2.putText(
                annotated_frame, 
                status_txt, 
                (40, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (240, 240, 240), 
                1, 
                cv2.LINE_AA
            )

            cv2.imshow(janela_nome, annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()