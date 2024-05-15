import mediapipe as mp 
import cv2
import numpy as np 
import imageio

mp_hands =  mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face =  mp.solutions.face_mesh
mp_pose =  mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence =  0.3, 
                        min_tracking_confidence =  0.3)

face = mp_face.FaceMesh(min_detection_confidence =  0.3, 
                        min_tracking_confidence =  0.3) 
hands =  mp_hands.Hands(min_detection_confidence =  0.3, 
                        min_tracking_confidence =  0.3)

cap  = cv2.VideoCapture('caida4.mp4')

# Lista para almacenar los frames del video
frames = []
frames2 = []
frame_count = 0

while True:
    ret, frame =  cap.read()
    if not ret:
        break
    results =  hands.process(frame)
    results_face =  face.process(frame)
    results_pose = pose.process(frame)
    points =  []
    height, width, _ = frame.shape
    img = np.zeros([height,width,3], dtype=np.uint8)
    img.fill(255)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                points.append([x,y,z])
            mp_drawing.draw_landmarks(img, hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1), 
                                 mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                                 )
            
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(img, face_landmarks,mp_face.FACEMESH_CONTOURS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1), 
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                                    )
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(img, results_pose.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1), 
                                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=1)
                                    )

    for hc in mp_hands.HAND_CONNECTIONS:
        print(hc)
        #v2.line(img,(int((points[hc[0]][0])* width),
         #             int((points[hc[0]][1]) * height)),
          #            (int((points[hc[1]][0])*width),
           #            int((points[hc[1]][1])*height)),(0,0,255),4)
    cv2.imshow('results', img)
    cv2.imshow('', frame)
    cv2.waitKey(1)
    
    # Agregar el frame actual a la lista de frames
    frames.append(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames2.append(frame)
    
    # Guardar el frame actual como una imagen
    cv2.imwrite(f'./img/frame_sk_{frame_count}.jpg', img)
    cv2.imwrite(f'./img/frame_{frame_count}.jpg', frame)
    frame_count += 1

# Guardar los frames como un gif
imageio.mimsave('./img/learning.gif', frames)
imageio.mimsave('./img/learning2.gif', frames2)