import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import shuffle
from keras import backend as K
K.set_image_dim_ordering('th')
import pickle
import sys


def rescale_frame(frame, percent=75):
    width = int(640)
    height = int(480)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

RESIZE_FACTOR = 4


class prediksi_video(object):  

    def from_class_to_label(self, kelas):
        data_path = os.path.expanduser('data_gambar_sklearn')
        data_dir_list = sorted(os.listdir(data_path))

        data_nama=[]
        for dataset in data_dir_list:
            img_list=os.listdir(data_path+'/'+ dataset)
            data_nama.append(dataset)

        list_labels = data_nama
        for i in list_labels:
            if kelas==list_labels.index(i):
                return i

    def show_video(self):
            video_capture = cv2.VideoCapture(0)
            
            nama_muka = []
            while True:
                ret, frame = video_capture.read()
                frame = rescale_frame(frame, percent=75)
                inImg = np.array(frame)
                outImg, nama_muka = self.process_image(inImg)
                cv2.imshow('Siaran Lansung :', outImg)

                # When everything is done, release the capture on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return
        
    def face1(self, face):
        num_channel = 1
        face_resized = cv2.resize(face, (112, 92))
        test_image = np.array(face_resized)
        test_image = test_image.astype('float64')
        test_image /= 255

        if num_channel==1:
            if K.image_dim_ordering()=='th':
                test_image= np.expand_dims(test_image, axis=0)
                test_image= np.expand_dims(test_image, axis=0)
            else:
                test_image= np.expand_dims(test_image, axis=3) 
                test_image= np.expand_dims(test_image, axis=0)
        else:
            if K.image_dim_ordering()=='th':
                test_image=np.rollaxis(test_image,2,0)
                test_image= np.expand_dims(test_image, axis=0)
            else:
                test_image= np.expand_dims(test_image, axis=0)
        return test_image

    def process_image(self, inImg):
        #================== LDA ================
       # with open ('model/LDA_mlp.pkl', 'rb') as Rpca:
           # model_1= pickle.load(Rpca)
        
       # with open ('model/model_LDA_mlp.pkl', 'rb') as Rca:
           # model_2= pickle.load(Rca) 
        #================== PCA=================    
        with open ('model/PCA_mlp.pkl', 'rb') as Rpca:
            model_1= pickle.load(Rpca)
        
        with open ('model/model_PCA_mlp.pkl', 'rb') as Rca:
            model_2= pickle.load(Rca) 
        #=====================Batas================
        RESIZE_FACTOR = 4
        global names
        frame = cv2.flip(inImg,1)
        resized_width, resized_height = (112, 92)
#
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized=cv2.resize(gray,(160, 130))
        

        cascPath = "haarcascade/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascPath)
        faces = face_cascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
        nama_prediksi = []
        for i in range(len(faces)):
            face_i = faces[i]
            x = face_i[0] * RESIZE_FACTOR
            y = face_i[1] * RESIZE_FACTOR
            w = face_i[2] * RESIZE_FACTOR
            h = face_i[3] * RESIZE_FACTOR
            face = gray[y:y+h, x:x+w]
            
            

            test_image1= self.face1(face)
            


            shape = test_image1.shape
            test_image1 = test_image1.reshape((shape[0], shape[1] * shape[2]*shape[3]))
            test_image1.shape

            hasil_pca=model_1.transform(test_image1)
            hasil_predik= model_2.predict(hasil_pca)[0]
                

            nama = self.from_class_to_label(hasil_predik)
            
            
            kn='unknown'
            if nama==kn:
                salah = 'Tidak Diketahui'
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame,salah, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

            else:
                
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(frame, (nama), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            nama_prediksi.append(nama)
        return (frame, nama_prediksi)

if __name__ == '__main__':
    prediksi = prediksi_video()
    print ("Press 'q' to quit video")
    prediksi.show_video()