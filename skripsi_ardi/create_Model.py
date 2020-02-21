import cv2
# print(cv2.__version__)
import numpy as np
import sys
import os
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from keras import backend as K
K.set_image_dim_ordering('th')

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def rescale_frame(frame, percent=75):
    width = int(640)
    height = int(480)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
# print('keras = 2.2.4')
nama_muka = sys.argv[1]

FREQ_DIV = 5   #frequency divider for capturing training images
RESIZE_FACTOR = 4
jmlh_training= 100
cascPath = "haarcascade/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
folder_umum = 'data_gambar_sklearn'
path = os.path.join(folder_umum, nama_muka)
if not os.path.isdir(path):
    os.mkdir(path)

count_captures = 0
count_timer = 0


class create_Model:
	 

	def ambil_gambar(self):
	    global count_timer
	    global count_captures 
	    video_capture=cv2.VideoCapture(0)
	    while True:
	        count_timer +=1
	        ret, frame = video_capture.read()
	        gambar_masuk=np.array(frame)
	        gambar_keluar= self.proses_gambar(gambar_masuk)
	        cv2.imshow('Siaran Lansung :',gambar_keluar)
	        if cv2.waitKey(1)&0xff==ord('q'):
	            video_capture.release()
	            cv2.destroyAllWindows()
	            return

	def proses_gambar(self, gambar_masuk):
	    global count_captures 
	    global count_timer
	    frame = cv2.flip(gambar_masuk,1)
	    resized_width, resized_height = (112, 92)
	    global jmlh_training  
	    if count_captures < jmlh_training:
	        frame = frame.astype(np.uint8)
	        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	        gray_resized = cv2.resize(gray, (160, 130))        
	        faces = face_cascade.detectMultiScale(
	            gray_resized,
	            scaleFactor=1.1,
	            minNeighbors=5,
	            minSize=(30, 30),
	            flags=cv2.CASCADE_SCALE_IMAGE
	            )
	        if len(faces) > 0:
	            areas = []
	            for (x, y, w, h) in faces: 
	                areas.append(w*h)
	            max_area, idx = max([(val,idx) for idx,val in enumerate(areas)])
	            face_sel = faces[idx]

	            x = face_sel[0] * RESIZE_FACTOR
	            y = face_sel[1] * RESIZE_FACTOR
	            w = face_sel[2] * RESIZE_FACTOR
	            h = face_sel[3] * RESIZE_FACTOR

	            face = gray[y:y+h, x:x+w]
	            face_resized = cv2.resize(face, (resized_width, resized_height))
	            img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(path) if fn[0]!='.' ]+[0])[-1] + 1

	            if count_timer%FREQ_DIV == 0:
	                cv2.imwrite('%s/%s.png' % (path, img_no), face_resized)
	                count_captures += 1
	                print ("gambar ke-: ", count_captures)

	            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
	            cv2.putText(frame,nama_muka, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0))
	    elif count_captures == jmlh_training:
	        count_captures += 1

	    return frame 

	def training_data(self):
	    data_path = os.path.expanduser('data_gambar_sklearn')
	    data_dir_list = sorted(os.listdir(data_path))
	    #pixel size
	    img_rows=112
	    img_cols=92
	    num_channel=1

	    #jumlah epoch
	    #num_epoch=3


	    # Define the number of classes
	    #num_classes = 8

	    img_data_list=[]
	    labels=[]
	    data_nama=[]
	    index = 0
	    for dataset in data_dir_list:
	        img_list=os.listdir(data_path+'/'+ dataset)
	        data_nama.append(dataset)
	        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	        for img in img_list:
	            input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
	            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	            input_img_resize=cv2.resize(input_img,(img_rows,img_cols))
	            img_data_list.append(input_img_resize)
	            label = index
	            labels.append(int(label))
	        index += 1
	    img_data = np.array(img_data_list)
	    img_data = img_data.astype('float64')
	    img_data /= 255
	    print (img_data.shape)

	    if num_channel==1:
	        if K.image_dim_ordering()=='th':
	            img_data= np.expand_dims(img_data, axis=1) 
	            print (img_data.shape)
	        else:
	            img_data= np.expand_dims(img_data, axis=4) 
	            print (img_data.shape)

	    else:
	        if K.image_dim_ordering()=='th':
	            img_data=np.rollaxis(img_data,3,1)
	            print (img_data.shape)

	    #%%
	    USE_SKLEARN_PREPROCESSING=False

	    if USE_SKLEARN_PREPROCESSING:
	        # using sklearn for preprocessing
	        from sklearn import preprocessing

	        def image_to_feature_vector(image, size=(img_rows, img_cols)):
	            # resize the image to a fixed size, then flatten the image into
	            # a list of raw pixel intensities
	            return cv2.resize(image, size).flatten()

	        img_data_list=[]
	        data_nama=[]    
	        for dataset in data_dir_list:
	            img_list=os.listdir(data_path+'/'+ dataset)
	            data_nama.append(dataset)
	            print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	            for img in img_list:
	                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
	                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	                input_img_flatten=image_to_feature_vector(input_img,(img_rows,img_cols))
	                img_data_list.append(input_img_flatten)
	                label = index
	                labels.append(int(label))
	            index += 1
	        img_data = np.array(img_data_list)
	        img_data = img_data.astype('float64')
	        print (img_data.shape)
	        img_data_scaled = preprocessing.scale(img_data)
	        print (img_data_scaled.shape)

	        print (np.mean(img_data_scaled))
	        print (np.std(img_data_scaled))

	        print (img_data_scaled.mean(axis=0))
	        print (img_data_scaled.std(axis=0))

	        if K.image_dim_ordering()=='th':
	            img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
	            print (img_data_scaled.shape)

	        else:
	            img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
	            print (img_data_scaled.shape)


	        if K.image_dim_ordering()=='th':
	            img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
	            print (img_data_scaled.shape)

	        else:
	            img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
	            print (img_data_scaled.shape)

	    if USE_SKLEARN_PREPROCESSING:
	        img_data=img_data_scaled
	    print(data_nama)
	    
	    shape = img_data.shape
	    img_data1 = img_data.reshape((shape[0], shape[1] * shape[2]*shape[3]))
	    img_data1.shape
	    x,y = shuffle(img_data1,labels, random_state=2)

	    # Split the dataset
	    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
        
        
	    #========================= PCA =============================================
# 	    n_components = 40

# 	    pca = PCA(n_components=n_components).fit(X_train)
# 	    # apply PCA transformation
# 	    #print(pca)

# 	    X_train_pca = pca.transform(X_train)
# 	    X_test_pca = pca.transform(X_test)

# 	    with open ('model/PCA_mlp.pkl', 'wb') as pca2:
# 	        pickle.dump(pca,pca2)
# 	    #MLP
# 	    #train a neural network
		
# 	    clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)
# 	    with open ('model/model_PCA_mlp.pkl', 'wb') as MLP:
# 	        pickle.dump(clf,MLP)
# 	    y_pred = clf.predict(X_test_pca)
# 	    print(y_pred)

# 	    akurasi=clf.score(X_test_pca, y_test) 
# 	    print(akurasi)
# 	    return (akurasi)
        
	  #========================= LDA =============================================
	    lda = LDA(n_components=40)
	    lda = lda.fit(X_train,y_train)
	    X_train_lda = lda.transform(X_train)
	    X_test_lda = lda.transform(X_test)

        #simpen Model PCD
	    with open ('model/LDA_mlp.pkl', 'wb') as lda2:
	        pickle.dump(lda,lda2)

        #Training using MPLClassifier LDA 
	    clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_lda, y_train)

        #save Model MLPC LDA
	    with open ('model/model_LDA_mlp.pkl', 'wb') as MLP:
	        pickle.dump(clf,MLP)
           
    
	    y_pred = clf.predict(X_test_lda)
	    print(y_pred)

	    akurasi=clf.score(X_test_lda, y_test) 
	    print(akurasi)
	    return (akurasi)
    #=========================== Batas =====================================
	    
if __name__ == '__main__':
	create_Model = create_Model()
	create_Model.ambil_gambar()
	create_Model.training_data()
	print ("Selesai bro")
	print ("Silahkan fitur Lainya")