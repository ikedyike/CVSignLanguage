import os
import pickle
import mediapipe as mp
import cv2 as cv
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

dataDirectory = './data'

data = []
labels = []

# Comments are visualization, must add [:1] after os.listdir inner for loop to visualize
for dir_ in os.listdir(dataDirectory):
    for img_path in os.listdir(os.path.join(dataDirectory, dir_)):
        data_aux = []

        img = cv.imread(os.path.join(dataDirectory, dir_, img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)


        results = hands.process(img_rgb)

        if (results.multi_hand_landmarks):
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_aux.append(x)
                    data_aux.append(y)

                # mp_drawing.draw_landmarks(
                #     img_rgb,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style(),
                # )
                    
            data.append(data_aux)
            labels.append(dir_)

        # plt.figure()
        # plt.imshow(img_rgb)

# plt.show()

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()