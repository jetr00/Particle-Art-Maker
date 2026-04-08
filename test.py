
import cv2
import numpy as np
import mediapipe as mp
import time

def draw_particles(image, detection_result, canvas, mask):
    canvas = canvas * 0.9
    canvas = canvas.astype(np.uint8)

    h, w, _ = image.shape

    for pose_landmarks in detection_result.pose_landmarks:
        for landmark in pose_landmarks:
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)
            cz = landmark.z
            cz_percentage = (cz - -1.0) / (1.0 - -1.0)
            mapped_cz = 10 + (cz_percentage * (2 - 10))
            mapped_cz = int(mapped_cz)
            mapped_cz = max(1, mapped_cz)

            cv2.circle(canvas, (cx, cy), mapped_cz, (255, 255, 255), -1)
        for pose_connections in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS:
            start_idx = pose_connections.start
            end_idx = pose_connections.end
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                start_landmark = pose_landmarks[start_idx]
                end_landmark = pose_landmarks[end_idx]

                start_z = start_landmark.z
                end_z = end_landmark.z
                z_percentage = ((start_z - -1.0) / (1.0 - -1.0) + (end_z - -1.0) / (1.0 - -1.0)) / 2
                mapped_z = 10 + (z_percentage * (2 - 10))
                mapped_z = int(mapped_z)
                mapped_z = max(1, mapped_z)

                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))

                cv2.line(canvas, start_point, end_point, (255, 255, 255), mapped_z, cv2.LINE_AA, 0)
                for i in range(100):
                    t = np.random.uniform(0.0, 1.0)
                    px = start_point[0] + t * (end_point[0] - start_point[0])
                    py = start_point[1] + t * (end_point[1] - start_point[1])
                    fx = px + np.random.randint(-15, 15)
                    fy = py + np.random.randint(-15, 15)
                    cv2.circle(canvas, (int(fx), int(fy)), 1, (255, 255, 255), -1, cv2.LINE_AA)
    xarray = np.random.randint(0, w-1, 20000)
    yarray = np.random.randint(0, h-1, 20000)

    value = mask[yarray, xarray]
    valid = (value < 0.5).flatten()
    
    final_x = xarray[valid]
    final_y = yarray[valid]
    
    canvas[final_y, final_x] = (255, 255, 255)
    canvas[np.clip(final_y + 1, 0, h-1), final_x] = (255, 255, 255)
    canvas = cv2.GaussianBlur(canvas, (7, 7), 0, 0, cv2.BORDER_DEFAULT)
    
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
        canvas = np.zeros_like(frame)
        art_frame = canvas

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
            
            cv2.imshow('AI Particle Art', art_frame)

            if cv2.waitKey(1) == ord('q'):
                break

video.release()
cv2.destroyAllWindows()
# Create a blank black canvas
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

particles = [] # To store [x, y, life, velocity_x, velocity_y]

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Flip for "mirror" effect and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. AI Processing
    results = pose.process(rgb_frame)

    # Fade the canvas slightly every frame (creates the "trail" effect)
    # The lower the 0.9, the shorter the trails.
    canvas = (canvas * 0.9).astype(np.uint8)

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            cx, cy = int(lm.x * w), int(lm.y * h)
            
            # 3. Spawn Particles at joints
            if lm.visibility > 0.5:
                # Add variation to make it look "organic" like your video
                for _ in range(2):
                    particles.append([
                        cx + random.randint(-5, 5), 
                        cy + random.randint(-5, 5), 
                        1.0, # Initial life (opacity)
                        random.uniform(-1, 1), # Velocity X
                        random.uniform(-2, 0)  # Velocity Y (drifting up)
                    ])

    # 4. Update and Draw Particles
    new_particles = []
    for p in particles:
        p[0] += p[3] # Move X
        p[1] += p[4] # Move Y
        p[2] -= 0.05 # Reduce life
        
        if p[2] > 0:
            # Draw a small circle for each particle
            brightness = int(255 * p[2])
            cv2.circle(canvas, (int(p[0]), int(p[1])), 1, (brightness, brightness, brightness), -1)
            new_particles.append(p)
    
    particles = new_particles

    # Optional: Draw the skeleton mesh lines
    mp_drawing.draw_landmarks(canvas, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=1))

    cv2.imshow('AI Particle Tracker', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()