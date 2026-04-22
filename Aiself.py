import cv2
import numpy as np

print("="*50)
print("SELF DRIVING CAR PERCEPTION")
print("="*50)

VIDEO_FILE = "./road.mp4"

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print(f"ERROR: Video '{VIDEO_FILE}' not found!")
    exit()

print(f"Video loaded: {VIDEO_FILE}")

def find_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (0, height//2),
        (width, height//2),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, 
                            minLineLength=40, maxLineGap=20)
    
    steering_angle = 0.0
    
    if lines is not None:
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 0.001)
            
            if slope < -0.3:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.3:
                right_lines.append((x1, y1, x2, y2))
        
        if left_lines:
            left_avg = np.mean(left_lines, axis=0).astype(int)
            cv2.line(frame, (left_avg[0], left_avg[1]), 
                    (left_avg[2], left_avg[3]), (0, 255, 0), 3)
        
        if right_lines:
            right_avg = np.mean(right_lines, axis=0).astype(int)
            cv2.line(frame, (right_avg[0], right_avg[1]), 
                    (right_avg[2], right_avg[3]), (0, 255, 0), 3)
        
        if left_lines and right_lines:
            center_x = width // 2
            left_x = np.mean([line[0] for line in left_lines] + 
                           [line[2] for line in left_lines])
            right_x = np.mean([line[0] for line in right_lines] + 
                            [line[2] for line in right_lines])
            lane_center = (left_x + right_x) / 2
            steering_angle = (lane_center - center_x) / center_x
            steering_angle = np.clip(steering_angle, -1, 1)
    
    return frame, steering_angle

def detect_potholes_on_road(frame, lane_lines_detected=True):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    road_mask = np.zeros((height, width), dtype=np.uint8)
    
    if lane_lines_detected:
        road_width = int(width * 0.6)
        road_left = (width - road_width) // 2
        road_right = road_left + road_width
        cv2.rectangle(road_mask, (road_left, height//2), (road_right, height), 255, -1)
    else:
        road_width = int(width * 0.7)
        road_left = (width - road_width) // 2
        cv2.rectangle(road_mask, (road_left, height//2), (road_right, height), 255, -1)
    
    # Detect dark spots
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Apply road mask - ONLY road area will be checked
    thresh_masked = cv2.bitwise_and(thresh, thresh, mask=road_mask)
    
    contours, _ = cv2.findContours(thresh_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potholes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Pothole size range
        if 200 < area < 4000:
            x, y, w, h = cv2.boundingRect(contour)
            # Check if contour is roughly circular/oval (pothole shape)
            aspect_ratio = w / (h + 0.01)
            if 0.5 < aspect_ratio < 2.0:  # Not too elongated
                potholes.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, "POTHOLE!", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame, len(potholes)

print("\nProcessing video...")
print("Controls: Q = Quit, Space = Pause")
print("="*50)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Total Frames: {total_frames}")

frame_count = 0
paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        
        if not ret:
            print("\nVideo finished!")
            break
        
        frame_count += 1
        
        # Lane detection
        frame_with_lanes, steering = find_lane(frame.copy())
        
        # Pothole detection - ONLY ON ROAD
        frame_with_potholes, pothole_count = detect_potholes_on_road(frame_with_lanes)
        
        # Show steering info
        if steering > 0.1:
            steer_text = "RIGHT"
            steer_color = (0, 0, 255)
        elif steering < -0.1:
            steer_text = "LEFT"
            steer_color = (0, 0, 255)
        else:
            steer_text = "STRAIGHT"
            steer_color = (0, 255, 0)
        
        cv2.putText(frame_with_potholes, f"STEERING: {steer_text}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, steer_color, 2)
        cv2.putText(frame_with_potholes, f"ANGLE: {steering:.2f}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame_with_potholes, f"FRAME: {frame_count}/{total_frames}", 
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        progress = (frame_count / total_frames) * 100
        cv2.putText(frame_with_potholes, f"PROGRESS: {progress:.1f}%", 
                    (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show pothole info ONLY if detected on road
        if pothole_count > 0:
            cv2.putText(frame_with_potholes, f"POTHOLE COUNT: {pothole_count}", 
                        (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_with_potholes, "SLOW DOWN!", 
                        (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("SELF DRIVING - LANE + POTHOLE DETECTION", frame_with_potholes)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("\nQuit!")
        break
    elif key == ord(' '):
        paused = not paused
        print("Paused" if paused else "Resumed")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("PROGRAM FINISHED!")
print("="*50)