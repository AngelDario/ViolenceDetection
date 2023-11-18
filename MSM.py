import cv2
import numpy as np

# Función para calcular el mapa de saliencia de movimiento
def calculate_motion_saliency(video_path, output_path):
    # Abre el video
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puedes cambiar el codec según tus necesidades
    output_video = cv2.VideoWriter( output_path, fourcc, fps, (width, height))

    # Lee el primer frame
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Inicializa el mapa de saliencia de movimiento
    motion_saliency_map = np.zeros_like(prvs, dtype=np.float32)

    while True:
        # Lee el siguiente frame
        ret, frame2 = cap.read()
        if not ret:
            break

        # Convierte los frames a escala de grises
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calcula el flujo óptico usando el método Farneback
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calcula la magnitud del flujo óptico
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Actualiza el mapa de saliencia de movimiento acumulando la magnitud del flujo
        motion_saliency_map += magnitude

        msm_frame = motion_saliency_map / np.max(motion_saliency_map)

        # Escribe el frame en el video de salida
        output_video.write(cv2.cvtColor(np.uint8(msm_frame * 255), cv2.COLOR_GRAY2BGR))

        # Actualiza el frame anterior
        prvs = next_frame

    # Libera los recursos
    cap.release()
    cv2.destroyAllWindows()

# # Ruta del archivo de video
# video_path = 'nuevo_video.avi'

# # Llama a la función para calcular el mapa de saliencia de movimiento
# def get_frame_count(video_path):
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Error al abrir el video.")
#         return -1

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     cap.release()
    
#     return frame_count

# print(get_frame_count(video_path))

