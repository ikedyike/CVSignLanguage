import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def getHandMove(hand_landmarks):
    landmarks = hand_landmarks.landmark

    if (landmarks[4].x < landmarks[3].x and all(landmarks[i].y > landmarks[i-2].y for i in range(8, 21, 4))):  # Other fingers are bent down
        return "ASL A"
    # Check for ASL "C"
    elif (landmarks[8].x < landmarks[6].x and landmarks[12].x < landmarks[10].x and landmarks[16].x < landmarks[14].x and landmarks[20].x < landmarks[18].x and landmarks[4].x > landmarks[3].x and landmarks[4].y < landmarks[3].y):
        return "ASL C"
    # Check for ASL "D"
    elif (landmarks[8].y < landmarks[6].y and all(landmarks[i].y > landmarks[i-2].y for i in range(12, 21, 4)) and landmarks[4].y < landmarks[2].y and landmarks[4].y > landmarks[3].y):
        return "ASL D"
    # Check for ASL "E"
    elif (landmarks[4].x > landmarks[3].x and all(landmarks[i].y > landmarks[i-2].y for i in range(8, 21, 4))):
        return "ASL E"
    # Check for ASL "F"
    elif (landmarks[4].x < landmarks[8].x and all(landmarks[i].y < landmarks[i-2].y for i in range(12, 21, 4))):
        return "ASL F"
    # Check for ASL "G"
    elif (landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[8].y < landmarks[7].y):
        return "ASL G"
    # Check for ASL "H"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and all(landmarks[i].y > landmarks[i-2].y for i in range(16, 21, 4))):
        return "ASL H"
    # Check for ASL "I"
    elif (landmarks[20].y < landmarks[18].y and all(landmarks[i].y > landmarks[i-2].y for i in range(8, 17, 4))):
        return "ASL I"
    # Check for ASL "J"
    elif (landmarks[20].y < landmarks[18].y and all(landmarks[i].y > landmarks[i-2].y for i in range(8, 17, 4)) and landmarks[20].x < landmarks[19].x):
        return "ASL J"
    # Check for ASL "K"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[4].y < landmarks[3].y and landmarks[4].x < landmarks[3].x):
        return "ASL K"
    # Check for ASL "L"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[4].x < landmarks[3].x):
        return "ASL L"
    # Check for ASL "M"
    elif (landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[8].x < landmarks[4].x < landmarks[3].x):
        return "ASL M"
    # Check for ASL "N"
    elif (landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[8].x < landmarks[4].x < landmarks[3].x):
        return "ASL N"
    # Check for ASL "O"
    elif (all(landmarks[i].y > landmarks[i-2].y for i in range(8, 21, 4)) and landmarks[4].x > landmarks[3].x and landmarks[4].y < landmarks[3].y):
        return "ASL O"
    # Check for ASL "P"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[4].x < landmarks[3].x):
        return "ASL P"
    # Check for ASL "Q"
    elif (landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[8].y < landmarks[7].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y):
        return "ASL Q"
    # Check for ASL "R"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[4].x > landmarks[3].x):
        return "ASL R"
    # Check for ASL "S"
    elif (all(landmarks[i].y > landmarks[i-2].y for i in range(8, 21, 4)) and landmarks[4].x > landmarks[3].x and landmarks[4].y > landmarks[3].y):
        return "ASL S"
    # Check for ASL "T"
    elif (landmarks[4].x > landmarks[3].x and all(landmarks[i].y > landmarks[i-2].y for i in range(8, 21, 4))):
        return "ASL T"
    # Check for ASL "U"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[4].x > landmarks[3].x):
        return "ASL U"
    # Check for ASL "V"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[4].x > landmarks[3].x):
        return "ASL V"
    # Check for ASL "W"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[4].x > landmarks[3].x):
        return "ASL W"
    # Check for ASL "X"
    elif (landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[4].x > landmarks[3].x):
        return "ASL X"
    # Check for ASL "Y"
    elif (landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y < landmarks[18].y and landmarks[4].x < landmarks[3].x):
        return "ASL Y"
    # Check for ASL "Z"
    elif (landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y and landmarks[4].x > landmarks[3].x and landmarks[8].x < landmarks[7].x):
        return "ASL Z"
    else: 
        return "Not ASL"

hand_move = None
gameText = ""

vid = cv.VideoCapture(0)

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = vid.read()
        if not ret or frame is None: 
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks_list = results.multi_hand_landmarks
            hand_landmarks = hand_landmarks_list[0]
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            hand_move = getHandMove(hand_landmarks)

        frame = cv.flip(frame, 1)
        gameText = f"Hand move = {hand_move}"
        cv.putText(frame, gameText, (25, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv.LINE_AA)
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'): 
            break
        
    vid.release()
    cv.destroyAllWindows()