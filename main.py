import cv2
import json
import time
import numpy as np
from ultralytics import YOLO
import supervision as sv
import serial

# ==========================================
# 1. CARREGAMENTO DO MODELO YOLO LOCAL
# ==========================================
MODEL_PATH = "runs/detect/treinamento_maquete/modelo_emergencia/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.50
TARGET_CLASS_NAME = "Emergencia"

print("Carregando modelo YOLO local...")
model = YOLO(MODEL_PATH)

# ==========================================
# 2. CONEXÃO SERIAL COM O ARDUINO
# ==========================================
ARDUINO_PORT = '/dev/ttyUSB0'  # Altere para 'COM3', 'COM4' se estiver no Windows
BAUD_RATE = 9600

arduino = None
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Conectado ao Arduino em {ARDUINO_PORT}")
except Exception as e:
    print(f"⚠️ Aviso: Arduino não conectado ({e}). Executando em modo simulação.")

# Guarda o último comando enviado para evitar envios repetidos na Serial
ultimo_comando_enviado = ""

def enviar_comando_arduino(comando):
    global ultimo_comando_enviado
    if comando != ultimo_comando_enviado:
        if arduino and arduino.is_open:
            arduino.write(f"{comando}\n".encode())
            print(f"--> [SERIAL] Comando enviado: {comando}")
        else:
            print(f"--> [SIMULAÇÃO] Comando enviado: {comando}")
        ultimo_comando_enviado = comando

# ==========================================
# 3. MARCAÇÃO DE REGIÕES (OpenCV Nativo)
# ==========================================
regions = {"norte": [], "sul": [], "leste": [], "oeste": []}
order = ["norte", "sul", "leste", "oeste"]
current_region_idx = 0
temp_points = []

def mouse_callback(event, x, y, flags, param):
    global temp_points
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_points.append((x, y))

def annotate_regions_local(cap):
    global current_region_idx, temp_points, regions
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Não foi possível acessar a webcam.")
        
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

        if key == 13 and len(temp_points) >= 3:  # Enter
            regions[reg_name] = temp_points.copy()
            temp_points = []
            current_region_idx += 1
        elif key == ord('c'):
            temp_points = []

    cv2.destroyWindow(window_name)

# ==========================================
# 4. CHECAGEM DE Bounding Box NAS REGIÕES
# ==========================================
def bb_region(box, regions, step=5):
    x1, y1, x2, y2 = map(int, box)
    for nome, pts in regions.items():
        if len(pts) == 0: continue
        polygon = np.array(pts, np.int32)
        for x in range(x1, x2 + 1, step):
            if cv2.pointPolygonTest(polygon, (float(x), float(y1)), False) >= 0 or \
               cv2.pointPolygonTest(polygon, (float(x), float(y2)), False) >= 0:
                return nome
        for y in range(y1, y2 + 1, step):
            if cv2.pointPolygonTest(polygon, (float(x1), float(y)), False) >= 0 or \
               cv2.pointPolygonTest(polygon, (float(x2), float(y)), False) >= 0:
                return nome
    return None

# ==========================================
# 5. INFERÊNCIA SLICING E LOOP PRINCIPAL
# ==========================================
SLICE_W, SLICE_H = 300, 300
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

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a webcam.")
        return

    annotate_regions_local(cap)
    
    mapa_comandos = {'norte': 'VN', 'sul': 'VS', 'leste': 'VL', 'oeste': 'VO'}

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            detections = slicer(frame)
            labels = []
            regioes_detectadas = []

            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                regiao = bb_region((x1, y1, x2, y2), regions)

                if regiao:
                    regioes_detectadas.append(regiao)
                    labels.append(f"{TARGET_CLASS_NAME} {detections.confidence[i]:.0%} | {regiao.upper()}")
                else:
                    labels.append(f"{TARGET_CLASS_NAME} {detections.confidence[i]:.0%}")

            # Lógica do Sinal Serial:
            if len(regioes_detectadas) > 0:
                # Pega a primeira via detectada com emergência
                primeira_regiao = regioes_detectadas[0]
                comando = mapa_comandos[primeira_regiao]
                enviar_comando_arduino(comando)
            else:
                # Nenhuma emergência na tela -> ciclo normal
                enviar_comando_arduino("XX")

            # Desenha overlay
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            colors_map = {"norte": (0, 0, 255), "sul": (255, 0, 0), "leste": (0, 255, 0), "oeste": (0, 165, 255)}
            for r_name, pts in regions.items():
                if len(pts) > 0:
                    cv2.polylines(annotated_frame, [np.array(pts, np.int32)], True, colors_map[r_name], 2)

            cv2.imshow("Sistema de Semáforo Inteligente", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if arduino and arduino.is_open:
            arduino.close()

if __name__ == "__main__":
    main()
