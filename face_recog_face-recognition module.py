import face_recognition
import cv2
import numpy as np
import os
import pickle
import time


def get_name():   # funtion to get name of person
    name=input("Enter your name:")
    return name

if (os.path.exists("face_images")==0):
    os.mkdir("face_images")

if os.path.exists('known_face_names.txt'):
    with open('known_face_names.txt','rb') as fp:
        known_face_names=pickle.load(fp)
else:
    known_face_names=[]


def cap_img(candidate_name):  # function to get image of person
    video_capture = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.putText(frame,"Press space to Capture",(0,20),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,0))
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256== 113:

            print("ending img capture")
            break
        elif k%256 == 32:

            img_name = os.path.join("face_images",candidate_name+".jpg")
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))


    video_capture.release()

    cv2.destroyAllWindows()


images={}
encoding={}

if os.path.exists('known_face_encodings.txt'):
    with open('known_face_encodings.txt','rb') as fp:
        known_face_encodings=pickle.load(fp)
else:
    known_face_encodings=[]

def make_encoding(candidate_name,known_face_encodings):
    images[candidate_name]=face_recognition.load_image_file(os.path.join("face_images",candidate_name+".jpg"))
    encoding[candidate_name]=face_recognition.face_encodings(images[candidate_name])[0]
    known_face_encodings.append(encoding[candidate_name])




def detect_faces_wc(known_face_encodings,known_face_names):
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    prev_frame_time = 0
    new_frame_time = 0

    while True:

        ret, frame = video_capture.read()


        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)


        rgb_small_frame = small_frame[:, :, ::-1]



        if process_this_frame:

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"


                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)

        fps = str(fps)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            #did image scaling to improve fps
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4


            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



        cv2.imshow('Video', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()
# code to execute programs in a closed manner
while True:
    choice = int(
        input("\n 1.Click Pic and encode \n 2.Webcam test \n"))
    if (choice == 1):
        candidate_name=get_name()
        known_face_names.append(candidate_name)
        cap_img(candidate_name)
        make_encoding(candidate_name,known_face_encodings)
    elif (choice == 2):
        detect_faces_wc(known_face_encodings,known_face_names)
    if (choice > 2):
        with open('known_face_names.txt', 'wb') as fp:
            pickle.dump(known_face_names,fp)
        with open('known_face_encodings.txt', 'wb') as fp:
            pickle.dump(known_face_encodings,fp)
        exit()