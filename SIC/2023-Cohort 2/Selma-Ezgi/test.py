
## TODO: Correct file paths!!!
##       plt.show yapilan kisimlarda yeni acilan sayfa kapatilmadan kodun devami run etmez
##       program kodunun tek bir giris noktasi (function) la calisir hale getirilmesi
##       webistesinded butona basilmasi halinde bu functionin kullanilmasi

import cv2
import matplotlib.pyplot as plt
import numpy as np
import gdown
import time
import zipfile
import os
import random
from tensorflow import keras
import tensorflow as tf
import pandas as pd 
from uuid import uuid4
# Add realife emotion update on the video?
#Training The Model
from tensorflow.keras import layers



def extract_files():
    # Veri seti dosyasının Google Drive'dan indirilmesi
    file_id = '1IXoLIw53F3B3ftVwaea_NCAw3TLWhzko'
    url = 'https://drive.google.com/uc?export=download&id=' + file_id
    output = 'dataset.zip'

    gdown.download(url, output, quiet=False)

    # Zip dosyasının çıkartılması
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset')

    # Çıkartılan dosyanın içeriğinin kontrolü
    extracted_files = os.listdir('dataset')
    print("Çıkartılan Dosyalar:", extracted_files)


def create_training_data(Class, Datadirectory, img_size):
    training_data = []
    for category in Class:
        path = os.path.join(Datadirectory,category)
        class_num = Class.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
    random.shuffle(training_data)
    training_data = random.sample(training_data, 7500) #downsampling for now
    X = []  #Data
    y = []  #Label

    for features,label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, img_size, img_size,3)    #convert it to 3D 
    X = X.astype(np.half)/255 # and normalize
    Y = np.array(y)

    return training_data, X, Y

def train_model():

    Class = ["0","1","2","3","4","5","6"]
    Datadirectory = "DATA/train"
    img_size = 224
    training_data, X, Y = create_training_data(Class, Datadirectory, img_size)
   

    model= tf.keras.applications.MobileNetV2()  #Pretrain the modeln
    model.summary()

    #Transfer Learning
    base_input= model.layers[0].input
    base_output=model.layers[-2].output

    #Adding layer and activate the functions
    final_output = layers.Dense(128)(base_output)
    final_output = layers.Activation("relu")(final_output)
    final_output = layers.Dense(64)(final_output)
    final_output = layers.Activation("relu")(final_output)
    final_output = layers.Dense(7,activation="softmax")(final_output)

    new_model=keras.Model(inputs=base_input,outputs=final_output)
    new_model.summary()
    new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    new_model.fit(X,Y,epochs = 3) # Less epoch for now

    new_model.save("my_model_64p07.h5") 

    return 0

def main():

    

    new_model=tf.keras.models.load_model("my_model_64p07.h5") #or we can fit it many way and ep 15, separate training to different function only train once and used the trained model 
    # new_model.evaluate #test data, we willl evaluate it as live image

    #RealTime Video Demo
    path = "haarscade_frontalface_default.xml"
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    rectangle_bgr = (255,255,255)
    img = np.zeros((500,500))
    text = "Random Text"

    (text_width,text_height) = cv2.getTextSize(text, font, fontScale = font_scale, thickness = 1)[0]
    text_offset_x = 10
    text_offset_y = img.shape[0]-25

    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img,box_coords[0], box_coords[1] ,rectangle_bgr, cv2.FILLED)
    cv2.putText(img,text,(text_offset_x, text_offset_y), font, fontScale=font_scale,color = (0,0,0), thickness = 1)


    cap=cv2.VideoCapture(0)
    if cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Can not open!")

    results = [] # pd.DataFrame(columns = ["frame", "frame_id", "frame_number", "face", "face_id", "face_number", "prediction", 'emotion'] )
    frame_number = 1
    
    while True:


        ret,frame = cap.read()

        cv2.imshow('frame', frame) 
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4) 
        
        frame_id = uuid4()
        face_number = 0
        
     
        for x,y,w,h in faces:
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
            faces_in_roi = faceCascade.detectMultiScale(roi_gray)
            # print(faces_in_roi)

            if len(faces_in_roi) == 0:
                print("Can not detect!")
            else:
                (ex, ey ,ew, eh) = faces_in_roi.flatten()
                
                face_roi = roi_color[ey:ey+eh, ex:ex+ew]
                final_image = cv2.resize(face_roi,(224,224)) 
                final_image = np.expand_dims(final_image,axis=0) #4D
                final_image = final_image /255.0 
                predictions = new_model.predict(final_image)
                emotion = emotion_assign(predictions) # Add information to panda data frame (frame number, face id, image, prediction etc .../ anything that might be useful)

                print("Success, Face detected!")
                face_number += 1
                face_id = uuid4()
                results.append([frame_id, frame_number, face_id, face_number, predictions, emotion])
                    # pd.DataFrame({"frame":frame, "frame_id":frame_id, 'frame_number':frame_number, 'face':face_roi, 'face_id':face_id, 'face_number':face_number, 'prediction':predictions, 'emotion':emotion}))

        
        frame_number += 1
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        import os
        
        if len(os.listdir("results")) > 2:
        # delete too many fıles
            import shutil
            shutil.rmtree("results")
            os.makedirs("results")
        
    cap.release()
    cv2.destroyAllWindows()
    results = pd.DataFrame(results, columns = ["frame_id", "frame_number", "face_id", "face_number", "prediction", 'emotion'])
    return results


def emotion_assign(predictions):

    emotion_dictionary = {"0":"Angry","1":"Disgust","2":"Fear","3":"Happy","4":"Neutral","5":"Sad","6":"Surprise"} # add printing on frame
    return emotion_dictionary.get(str(np.argmax(predictions)))
    


        
        

        
if __name__ == "__main__" :
   # train_model()
   results = main()
   print(results.head())
   results.to_csv("results.tsv", sep = "\t") 