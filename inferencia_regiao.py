import cv2
import json
import time
import numpy as np
from ultralytics import YOLO
import supervision as sv
import serial  # Comunicação Serial com o Arduino

# ==========================================
# 1. CONFIGURAÇÃO DO MODELO LOCAL YOLO
# ==========================================
# Caminho para os pesos do seu modelo treinado
MODEL_PATH = "runs/detect/treinamento_maquete/modelo_emergencia/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.50
TARGET_CLASS_NAME = "Emergencia"

print("Carregando modelo YOLO local...")
model = YOLO(MODEL_PATH)

# ==========================================
# 2. CONFIGURAÇÃO DA COMUNICAÇÃO SERIAL (ARDUINO)
# ==========================================
# No Linux costuma ser '/dev/ttyUSB0' ou '/dev/ttyACM0'. No Windows 'COM3', 'COM4', etc.
ARDUINO_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 9600

arduino = None
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Aguarda tempo de reinicialização da serial do Arduino
    print(f"Conectado ao Arduino com sucesso na porta {ARDUINO_PORT}!")
except Exception as e:
    print(f"⚠️ Aviso: Não foi possível conectar ao Arduino ({e}). O código executará em modo simulação.")

# ==========================================
# 3. MARCAÇÃO INTERATIVA DAS REGIÕES (OpenCV Nativo)
# ==========================================
regions = {
    "norte": [],
    "sul": [],
    "leste": [],
    "oeste": []
}

current_region_idx = 0
order = ["norte", "sul", "leste", "oeste"]
temp_points = []

def mouse_callback(event, x, y, flags, param):
    global temp_points
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_points.append((x, y))

def annotate_regions_local(cap):
    global current_region_idx, temp_points, regions
    
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Não foi possível capturar o frame da webcam.")
        
    window_name = "Marcacao de Regioes (Clique nos pontos)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    colors = {
        "norte": (0, 0, 255),    # Vermelho
        "sul": (255, 0, 0),      # Azul
        "leste": (0, 255, 0),    # Verde
        "oeste": (0, 165, 255)   # Laranja
    }

    while current_region_idx < len(order):
        reg_name = order[current_region_idx]
        canvas = frame.copy()

        # Desenha regiões já salvas
        for r_name, pts in regions.items():
            if len(pts) > 0:
                poly = np.array(pts, np.int32)
                cv2.polylines(canvas, [poly], isClosed=True, color=colors[r_name], thickness=2)

        # Desenha pontos atuais da região em marcação
        if len(temp_points) > 0:
            for pt in temp_points:
                cv2.circle(canvas, pt, 4, colors[reg_name], -1)
            if len(temp_points) > 1:
                pts_arr = np.array(temp_points, np.int32)
                cv2.polylines(canvas, [pts_arr], isClosed=False, color=colors[reg_name], thickness=2)

        # Instruções na tela
        cv2.putText(canvas, f"Desenhe a regiao: {reg_name.upper()}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Pressione 'ENTER' para confirmar | 'C' para limpar a regiao", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Tecla Enter
            if len(temp_points) >= 3:
                regions[reg_name] = temp_points.copy()
                temp_points = []
                current_region_idx += 1
            else:
                print("⚠️ Marque pelo menos 3 pontos para formar o polígono da região.")
        elif key == ord('c'):  # Tecla C
            temp_points = []

    cv2.destroyWindow(window_name)
    print(" Regiões configuradas:")
    print(json.dumps(regions, indent=4))

# ==========================================
# 4. FUNÇÃO DE INTERSECÇÃO DA BB COM AS REGIÕES
# ==========================================
def bb_region(box, regions, step=5):
    x1, y1, x2, y2 = map(int, box)

    for nome, pts in regions.items():
        if len(pts) == 0:
            continue
        polygon = np.array(pts, np.int32)

        # Checa borda superior e inferior
        for x in range(x1, x2 + 1, step):
            if cv2.pointPolygonTest(polygon, (float(x), float(y1)), False) >= 0:
                return nome
            if cv2.pointPolygonTest(polygon, (float(x), float(y2)), False) >= 0:
                return nome

        # Checa bordas laterais
        for y in range(y1, y2 + 1, step):
            if cv2.pointPolygonTest(polygon, (float(x1), float(y)), False) >= 0:
                return nome
            if cv2.pointPolygonTest(polygon, (float(x2), float(y)), False) >= 0:
                return nome

    return None

# ==========================================
# 5. INFERÊNCIA COM FATIAMENTO (Slicing) LOCAL
# ==========================================
SLICE_W, SLICE_H = 300, 300
OVERLAP_PERCENTAGE = 0.4
OVERLAP_W = int(SLICE_W * OVERLAP_PERCENTAGE)
OVERLAP_H = int(SLICE_H * OVERLAP_PERCENTAGE)

def predict_slice(image_slice: np.ndarray) -> sv.Detections:
    # Executa a predição no modelo YOLO local
    results = model(image_slice, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    if len(detections) > 0:
        conf_mask = detections.confidence > CONFIDENCE_THRESHOLD
        detections = detections[conf_mask]

    return detections

slicer = sv.InferenceSlicer(
    callback=predict_slice,
    slice_wh=(SLICE_W, SLICE_H),
    overlap_wh=(OVERLAP_W, OVERLAP_H)
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# ==========================================
# 6. ENVIO DE COMANDO VIA SERIAL
# ==========================================
def enviar_comando_arduino(regiao):
    if arduino and arduino.is_open:
        # Mapeia cada região para um caractere enviado via Serial ('N', 'S', 'L', 'O')
        mapa_comandos = {'norte': 'N', 'sul': 'S', 'leste': 'L', 'oeste': 'O'}
        cmd = mapa_comandos.get(regiao)
        if cmd:
            arduino.write(cmd.encode())
            print(f"--> [ARDUINO] Comando '{cmd}' enviado! (Abrir Sinal {regiao.upper()})")

# ==========================================
# 7. LOOP PRINCIPAL
# ==========================================
def main():
    cap = cv2.VideoCapture(0)  # 0 para a webcam principal
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    # Passo 1: Marcação inicial das regiões na webcam
    annotate_regions_local(cap)

    print("\nIniciando monitoramento do cruzamento. Pressione 'q' na janela para sair.\n")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue  # Processa frames alternados para otimizar desempenho

            # Inferência com fatiamento de imagem
            detections = slicer(frame)
            labels = []
            regioes_com_emergencia = set()

            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                regiao = bb_region((x1, y1, x2, y2), regions)

                if regiao is not None:
                    print(f"🚨 Veículo de emergência na via: {regiao.upper()}")
                    labels.append(f"{TARGET_CLASS_NAME} {detections.confidence[i]:.0%} | {regiao.upper()}")
                    regioes_com_emergencia.add(regiao)
                else:
                    labels.append(f"{TARGET_CLASS_NAME} {detections.confidence[i]:.0%}")

            # Envia o sinal do Arduino para as vias que possuem veículos detectados
            for reg in regioes_com_emergencia:
                enviar_comando_arduino(reg)

            # Desenha as caixas e rótulos no frame
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            # Desenha os polígonos das vias no frame
            colors_map = {"norte": (0, 0, 255), "sul": (255, 0, 0), "leste": (0, 255, 0), "oeste": (0, 165, 255)}
            for r_name, pts in regions.items():
                if len(pts) > 0:
                    cv2.polylines(annotated_frame, [np.array(pts, np.int32)], isClosed=True, color=colors_map[r_name], thickness=2)

            # Exibe o resultado na janela OpenCV
            cv2.imshow("Sistema de Semaforo Inteligente - YOLO", annotated_frame)

            # Pressione 'q' para encerrar a execução
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arduino and arduino.is_open:
            arduino.close()
        print("\nSistema finalizado.")

if __name__ == "__main__":
    main()
