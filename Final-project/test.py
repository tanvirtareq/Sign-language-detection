from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np



import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp



class Test:
    def __init__(self, obj):
        self.obj=obj
        
        # Path for exported data, numpy arrays
        self.DATA_PATH = os.path.join('MP_Data') 

        # Actions that we try to detect
        self.actions = np.array(['অ', 'ই', 'গ'])

        # Thirty videos worth of data
        self.no_sequences = 30

        # Videos are going to be 30 frames in length
        self.sequence_length = 30

        # Folder start
        self.start_folder = 30

        self.mp_holistic = mp.solutions.holistic # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities

        
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.model.load_weights('Final-project/action.h5')

        self.mapping={'অ':'O', 'ই':'E', 'গ':'Go'}
        self.reverse_mapping={'O':'অ', 'E':'ই', 'Go':'গ'}

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def draw_landmarks(self, image, results):
    #     mp_drawing.draw_landmarks(image, results.face_landmarks) # Draw face connections
    #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
    def draw_styled_landmarks(self, image, results):
    #     # Draw face connections
    #     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
    #                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                              ) 
    #     # Draw pose connections
    #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
    #                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
    #                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    #                              ) 
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #     return np.concatenate([pose, face, lh, rh])
        return np.concatenate([lh, rh])

    def start(self):
        print("thikase")
        # 1. New detection variables

        import cv2

        self.sequence = []
        self.sentence = []
        self.threshold = 0.8

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        # res=[0, 0, 0]
        self.res=[]
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = self.mediapipe_detection(frame, holistic)
        #         print(results)
                
                # Draw landmarks
                self.draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = self.extract_keypoints(results)
                sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
        #         sequence.append(keypoints)
                sequence = sequence[:self.sequence_length]
                
                if len(sequence) == self.sequence_length:
                    res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
        #             print(res)
        #             print(actions[np.argmax(res)])
                    
                #3. Viz logic
                if len(res)>0 and res[np.argmax(res)] > self.threshold: 
        #             print(actions[np.argmax(res)])
                    if len(sentence) > 0: 
                        if self.actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(self.mapping[self.actions[np.argmax(res)]])
                    else:
                        sentence.append(self.mapping[self.actions[np.argmax(res)]])
        #         print(sentence)

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                    # Viz probabilities
        #             image = prob_viz(res, actions, image, colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
