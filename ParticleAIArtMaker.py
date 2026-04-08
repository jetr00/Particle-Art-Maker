import cv2
import numpy as np
import mediapipe as mp
import time

def draw_particles(image, detection_result, canvas, mask):
    h, w, _ = image.shape

    if not detection_result.pose_landmarks: return canvas
    
    landmarkers = detection_result.pose_landmarks[0]
    center_x = (landmarkers[11].x + landmarkers[12].x) / 2 * w
    center_y = (landmarkers[11].y + landmarkers[12].y) / 2 * h

    zoom = cv2.getRotationMatrix2D((center_x, center_y), 0, 1.02)
    canvas = cv2.warpAffine(canvas, zoom,(w, h))
    canvas = (canvas * 0.95).astype(np.uint8)

    xarray = np.random.randint(0, w-1, 80000)
    yarray = np.random.randint(0, h-1, 80000)

    value = mask[yarray, xarray]
    valid = (value < 0.5).flatten()
    
    final_x = xarray[valid]
    final_y = yarray[valid]

    canvas[final_y, final_x] = (255, 255, 255)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    
    return canvas

model_path = 'pose_landmarker_lite.task' 
image_model_path = 'selfie_segmenter.tflite'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
    )

ImageOptions = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=image_model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_category_mask=True
    )

video = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    with ImageSegmenter.create_from_options(ImageOptions) as segmenter:
        if not video.isOpened():
            print("Camera Unavailable")
            exit()

        start_time = time.time()
        
        ret, frame = video.read()
        if not ret: exit()
        art_frame = np.zeros_like(frame)

        while True:
            ret, frame = video.read()
            if not ret: break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.time() - start_time) * 1000)

            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            silouette_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            sillouette_array = silouette_result.category_mask.numpy_view()

            art_frame = draw_particles(frame, pose_landmarker_result, art_frame, sillouette_array)
            
            window = cv2.resize(art_frame, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.imshow('AI Particle Art', window)

            if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord(';'):
                break

video.release()
cv2.destroyAllWindows()