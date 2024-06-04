import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

vid = cv.VideoCapture(0)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'NEXT', 27: 'END'}

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

display_text = ""

instruction_text = "Show NEXT symbol to begin spelling a word."

nextChar = False
startWord = True
nextCharTime = 0

while True:

    data_aux = []

    ret, frame = vid.read()

    # Add a rectangle on the left side of the frame
    start_point = (50, 100) 
    end_point = (250, 400)
    color = (203, 65, 84)
    thickness = 3

    cv.rectangle(frame, start_point, end_point, color, thickness)

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())
        
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            data_aux.append(x)
            data_aux.append(y)

        prediction = model.predict([np.asarray(data_aux)])

        predictedChar = labels_dict[int(prediction[0])]

        if predictedChar == 'END':
            display_text = ""
            nextChar = False
            startWord = True
        elif predictedChar == 'NEXT':
            nextChar = True
            startWord = False
            nextCharTime = time.time() + 3  # 3 seconds delay
        elif nextChar and time.time() > nextCharTime:
            if predictedChar != 'END' and predictedChar != 'NEXT':
                display_text += predictedChar
                nextChar = False

    # Update instruction text with countdown timer if nextChar is True
    if nextChar:
        remaining_time = max(0, int(nextCharTime - time.time()))
        if remaining_time > 0:
            instruction_text = f"{remaining_time}..."
        else:
            instruction_text = ""
    else:
        if startWord:
            instruction_text = "Show NEXT symbol to begin spelling a word."
        else:
            instruction_text = "Show NEXT symbol for next letter."

    # Display the instruction text on the frame, two layers to imitate "Outlined" image.
    instruction_text_width, _ = cv.getTextSize(instruction_text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 6)[0]
    instruction_text_x = int((frame.shape[1] - instruction_text_width) / 2)
    instruction_text_y = 50
    cv.putText(frame, instruction_text, (instruction_text_x, instruction_text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv.LINE_AA)

    instruction_text_width, _ = cv.getTextSize(instruction_text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    instruction_text_x = int((frame.shape[1] - instruction_text_width) / 2)
    instruction_text_y = 50
    cv.putText(frame, instruction_text, (instruction_text_x, instruction_text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    # Display the text on the frame, two layers to imitate "Outlined" image
    text_width, _ = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 1, 6)[0]
    text_x = int((frame.shape[1] - text_width) / 2)
    text_y = frame.shape[0] - 10
    cv.putText(frame, display_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv.LINE_AA)

    text_width, _ = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = int((frame.shape[1] - text_width) / 2)
    text_y = frame.shape[0] - 10
    cv.putText(frame, display_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow('frame', frame)
    cv.waitKey(1)
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
        
vid.release()
cv.destroyAllWindows()
