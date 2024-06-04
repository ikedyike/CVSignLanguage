import os
import cv2 as cv

dataDirectory = './data'
if not os.path.exists(dataDirectory):
    os.makedirs(dataDirectory)

number_of_classes = 28
datasetSize = 200

# cam = cv.VideoCapture(0)
# for i in range(number_of_classes):
#     class_directory = os.path.join(dataDirectory, '{:02d}'.format(i))
#     if not os.path.exists(class_directory):
#         os.makedirs(class_directory)
    
#     print('Collecting data for class {}'.format(i))

#     done = False
#     while True:
#         ret, frame = cam.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
    
#         # Add a rectangle on the left side of the frame
#         start_point = (50, 100) 
#         end_point = (250, 400)
#         color = (203, 65, 84)
#         thickness = 3

#         cv.rectangle(frame, start_point, end_point, color, thickness)

#         cv.putText(frame, 'Ready? Press R!', (25, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5, cv.LINE_AA)
#         cv.putText(frame, 'Ready? Press R!', (25, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv.LINE_AA)
#         cv.imshow('frame', frame)
#         if cv.waitKey(25) == ord('r'):
#             break

#     counter = 0
#     while counter < datasetSize:
#         ret, frame = cam.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         cv.rectangle(frame, start_point, end_point, color, thickness)

#         cv.imshow('frame', frame)
#         cv.waitKey(25)
#         cv.imwrite(os.path.join(class_directory, '{}.jpg'.format(counter)), frame)
#         counter += 1

# cam.release()
# cv.destroyAllWindows()

# Class number hardcoded
class_to_collect = 24

# Validate the class number
if class_to_collect < 0 or class_to_collect >= number_of_classes:
    print(f"Invalid class number. Please enter a number between 0 and {number_of_classes-1}.")
else:
    class_directory = os.path.join(dataDirectory, '{:02d}'.format(class_to_collect))
    if not os.path.exists(class_directory):
        os.makedirs(class_directory)

    print(f'Collecting data for class {class_to_collect}')

    cam = cv.VideoCapture(0)
    done = False
    while True:
        ret, frame = cam.read()

        # Add a rectangle on the left side of the frame
        start_point = (50, 100) 
        end_point = (250, 400)
        color = (203, 65, 84)
        thickness = 3

        cv.rectangle(frame, start_point, end_point, color, thickness)

        cv.putText(frame, 'Ready? Press R!', (25, 80), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5, cv.LINE_AA)
        cv.putText(frame, 'Ready? Press R!', (25, 80), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow('frame', frame)
        if cv.waitKey(25) == ord('r'):
            break

    counter = 0
    while counter < datasetSize:
        ret, frame = cam.read()

        # Add a rectangle on the left side of the frame
        start_point = (50, 100) 
        end_point = (250, 400)
        color = (203, 65, 84)
        thickness = 3

        cv.rectangle(frame, start_point, end_point, color, thickness)

        cv.imshow('frame', frame)
        cv.waitKey(25)
        cv.imwrite(os.path.join(class_directory, '{}.jpg'.format(counter)), frame)
        counter += 1

    cam.release()
    cv.destroyAllWindows()