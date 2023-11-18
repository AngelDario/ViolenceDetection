import cv2
import numpy as np

def calculate_motion_saliency(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    output_video = cv2.VideoWriter( output_path, fourcc, fps, (width, height))

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    motion_saliency_map = np.zeros_like(prvs, dtype=np.float32)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        motion_saliency_map += magnitude

        msm_frame = motion_saliency_map / np.max(motion_saliency_map)

        output_video.write(cv2.cvtColor(np.uint8(msm_frame * 255), cv2.COLOR_GRAY2BGR))

        prvs = next_frame

    cap.release()
    cv2.destroyAllWindows()
