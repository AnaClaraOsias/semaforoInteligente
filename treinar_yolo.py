from ultralytics import YOLO

# 1. Carrega a arquitetura da YOLO11
# Usaremos a versão 'small' (yolo11s.pt), que é um pouco mais pesada que a nano, 
# mas muito mais precisa para os pequenos detalhes da sua maquete.
model = YOLO('yolo11s.pt')

# 2. Inicia o Fine-Tuning
print("Iniciando o treinamento na RTX 4090...")

resultados = model.train(
    data='./dataset_maquete/data.yaml', # Onde estão as imagens e classes
    epochs=100,                         # Quantas vezes a IA vai ver o dataset inteiro
    imgsz=640,                          # Resolução das imagens
    batch=32,                           # Como você tem 24GB de VRAM, podemos processar 32 imagens por vez
    device=0,                           # Trava o treinamento na GPU 0
    project='treinamento_maquete',      # Cria uma pasta principal para salvar os resultados
    name='modelo_emergencia',           # Nome da subpasta deste experimento
    patience=20                         # Se a IA parar de aprender por 20 épocas seguidas, ela encerra mais cedo
)

print("Treinamento finalizado com sucesso!")
