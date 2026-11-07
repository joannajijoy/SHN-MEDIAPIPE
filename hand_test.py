import cv2
import mediapipe as mp
import time

# 1. Initialise MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 2. Configure the hand detector
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# 3. Hand landmark connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]
def is_pointing(hand):
    index_up = hand[8].y < hand[6].y

    middle_folded = hand[12].y > hand[10].y
    ring_folded = hand[16].y > hand[14].y
    pinky_folded = hand[20].y > hand[18].y

    return index_up and middle_folded and ring_folded and pinky_folded

# 4. Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam!")
    exit()

# 5. Create the Hand Landmarker
with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            print("Failed to capture frame")
            break

        # Flip webcam image
        frame = cv2.flip(frame, 1)

        # Get image dimensions
        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate timestamp in milliseconds
        timestamp = time.monotonic_ns() // 1_000_000

        # Detect hand landmarks
        results = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # 6. Draw detected hands
        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                # Draw connections
                for start, end in connections:

                    x1 = int(hand_landmarks[start].x * width)
                    y1 = int(hand_landmarks[start].y * height)

                    x2 = int(hand_landmarks[end].x * width)
                    y2 = int(hand_landmarks[end].y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                # Draw landmark points
                for landmark in hand_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 0, 255),
                        -1
                    )
                if is_pointing(hand_landmarks):
    cv2.putText(
        frame,
        "POINTING",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

        # Display webcam
        cv2.imshow("GardenHands - Hand Tracking", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 7. Release webcam
cap.release()
cv2.destroyAllWindows()