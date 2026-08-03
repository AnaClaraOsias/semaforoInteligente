import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import json
import time
import numpy as np
from ultralytics import YOLO
import supervision as sv
import serial

# ==========================================
# 1. CONFIGURAÇÕES E CARREGAMENTO DO MODELO
# ==========================================
MODEL_PATH = "runs/detect/treinamento_maquete/modelo_emergencia-2/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.50
TARGET_CLASS_NAME = "Emergencia"
REGIOES_FILE = "regioes_semaforo.json"

# CONFIGURAÇÕES DE DESEMPENHO E OTIMIZAÇÃO
USE_SLICING = False       
FRAME_STRIDE = 3         # Descarta (FRAME_STRIDE - 1) quadros para manter tempo real

print(f"Carregando modelo YOLO local (Slicing: {USE_SLICING} | Stride: {FRAME_STRIDE})...")
model = YOLO(MODEL_PATH)

# ==========================================
# 2. CONEXÃO SERIAL REAL COM O ARDUINO
# ==========================================
SERIAL_PORT = '/dev/ttyACM0'  # Altere para '/dev/ttyUSB0' se necessário
BAUD_RATE = 9600

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Tempo para reset do Arduino ao conectar
    print(f"✅ Conectado com sucesso ao Arduino em {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ Não foi possível abrir a porta {SERIAL_PORT}: {e}")
    print("   O código continuará rodando em MODO SIMULAÇÃO.")
    arduino = None

ultimo_comando_enviado = ""

def enviar_comando_arduino(comando):
    global ultimo_comando_enviado, arduino
    if comando != ultimo_comando_enviado:
        print(f"--> [SERIAL REAL] Enviando comando: {comando}")
        
        if arduino and arduino.is_open:
            mensagem = f"{comando}\n"
            arduino.write(mensagem.encode('utf-8'))
            
        ultimo_comando_enviado = comando

# ==========================================
# 3. MARCAÇÃO E MANIPULAÇÃO DE REGIÕES
# ==========================================
regions = {"norte": [], "sul": [], "leste": [], "oeste": []}
order = ["norte", "sul", "leste", "oeste"]

def salvar_regioes(caminho, regioes):
    with open(caminho, "w") as f:
        json.dump(regioes, f, indent=4)
    print(f"✅ Regiões salvas com sucesso em '{caminho}'.")

def carregar_regioes(caminho):
    with open(caminho, "r") as f:
        regioes = json.load(f)
    print(f"📂 Regiões carregadas de '{caminho}'.")
    return regioes

def mouse_callback(event, x, y, flags, param):
    global temp_points
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_points.append((x, y))

temp_points = []
def annotate_regions_local(cap):
    global temp_points, regions
    current_region_idx = 0
    temp_points = []
    regions = {"norte": [], "sul": [], "leste": [], "oeste": []}

    ret, frame = cap.read()
    if not ret: raise RuntimeError("Não foi possível acessar a webcam.")
        
    window_name = "Marcacao de Regioes (Clique nos pontos)"
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

        cv2.putText(canvas, f"Desenhe a regiao: {reg_name.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
# 4. DECISÃO DE REGIÃO POR MAIOR INTERSECÇÃO
# ==========================================
def bb_best_region(box, regions):
    x1, y1, x2, y2 = map(int, box)
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)

    best_region = None
    max_intersection_area = 0

    for nome, pts in regions.items():
        if len(pts) < 3: continue

        pts_local = np.array(pts, dtype=np.int32) - np.array([x1, y1])
        mask = np.zeros((box_h, box_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_local], 255)

        intersection_area = cv2.countNonZero(mask)

        if intersection_area > max_intersection_area:
            max_intersection_area = intersection_area
            best_region = nome

    return best_region if max_intersection_area > 0 else None

# ==========================================
# 5. LÓGICA DE INFERÊNCIA
# ==========================================
SLICE_W, SLICE_H = 320, 320
OVERLAP_PERCENTAGE = 0.4

def predict_slice(image_slice: np.ndarray) -> sv.Detections:
    results = model(image_slice, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(detections) > 0:
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    return detections

slicer = sv.InferenceSlicer(
    callback=predict_slice,
    slice_wh=(SLICE_W, SLICE_H),
    overlap_wh=(int(SLICE_W * OVERLAP_PERCENTAGE), int(SLICE_H * OVERLAP_PERCENTAGE))
)

def executar_inferencia(frame):
    if USE_SLICING:
        return slicer(frame)
    else:
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
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
        resposta = input(f"\nJá existe um arquivo '{REGIOES_FILE}'. Deseja reanotar as regiões? [s/N]: ").strip().lower()
        if resposta == 's':
            cap_temp = cv2.VideoCapture(2)
            if not cap_temp.isOpened(): return
            annotate_regions_local(cap_temp)
            salvar_regioes(REGIOES_FILE, regions)
            cap_temp.release()
        else:
            regions = carregar_regioes(REGIOES_FILE)
    else:
        cap_temp = cv2.VideoCapture(2)
        if not cap_temp.isOpened(): return
        annotate_regions_local(cap_temp)
        salvar_regioes(REGIOES_FILE, regions)
        cap_temp.release()

    cap = cv2.VideoCapture(2)
    if not cap.isOpened(): return

    # Reduz o buffer interno sem alterar a resolução nativa da câmera
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Aguardando estabilização da câmera...")
    time.sleep(1.5) 
    for _ in range(5): cap.read()

    mapa_comandos = {'norte': 'VN', 'sul': 'VS', 'leste': 'VL', 'oeste': 'VO'}

    janela_nome = "Sistema de Semaforo Inteligente"
    cv2.namedWindow(janela_nome)
    cv2.waitKey(1)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            # Descarte de quadros acumulados no buffer para garantir imagem em tempo real
            for _ in range(FRAME_STRIDE - 1):
                cap.grab()

            ret, frame = cap.read()
            if not ret: break

            # 1. CÁLCULO DE FPS
            curr_time = time.time()
            elapsed_time = curr_time - prev_time
            prev_time = curr_time
            if elapsed_time > 0: fps = 1.0 / elapsed_time

            # 2. INFERÊNCIA E CLASSIFICAÇÃO IMEDIATA
            detections = executar_inferencia(frame)
            labels = []
            regioes_frame_atual = []

            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                regiao = bb_best_region((x1, y1, x2, y2), regions)

                if regiao:
                    regioes_frame_atual.append(regiao)
                    labels.append(f"{TARGET_CLASS_NAME} {detections.confidence[i]:.0%} | {regiao.upper()}")
                else:
                    labels.append(f"{TARGET_CLASS_NAME} {detections.confidence[i]:.0%}")

            # Decisão instantânea do quadro atual (sem votação)
            if len(regioes_frame_atual) > 0:
                regiao_confirmada = regioes_frame_atual[0]
                comando_decidido = mapa_comandos[regiao_confirmada]
            else:
                regiao_confirmada = None
                comando_decidido = "XX"

            # Envia para a Serial (somente se houver alteração de estado)
            enviar_comando_arduino(comando_decidido)

            # 3. DESENHO DA INTERFACE
            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            fps_str = f"FPS: {fps:.0f}"
            
            if regiao_confirmada:
                status_txt = f"EMERGENCIA: {regiao_confirmada.upper()} ({comando_decidido}) | {fps_str}"
                cor_indicador = (0, 0, 255)  # Vermelho
            else:
                status_txt = f"NORMAL ({comando_decidido}) | {fps_str}"
                cor_indicador = (0, 255, 0)  # Verde

            # Badge superior
            cv2.rectangle(annotated_frame, (10, 10), (360, 40), (20, 20, 20), -1)
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
        if arduino and arduino.is_open:
            arduino.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
