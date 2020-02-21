from tkinter import *

import os

root = Tk(className = 'EigenFaces_gui_Ardi')

canvas= Canvas(root, width=600, height=300)
canvas.pack()
svalue = StringVar() # defines the widget state as string


input_nama = Entry(root,textvariable=svalue) # adds a textarea widget
canvas.create_window(170,210, window=input_nama)

label1 = Label(root, text= 'Eigenfaces PCA')
label1.config(font=('helvetica', 35))
canvas.create_window(300,50, window=label1)

label2 = Label(root, text= 'Nama :')
label2.config(font=('helvetica', 10))
canvas.create_window(70,210, window=label2)


def train_eigen_btn_load():
    name = svalue.get()
    os.system('python create_model.py %s'%name)



def recog_eigen_btn_load():
    os.system('python prediksi_using_model.py')


trainE_btn = Button(root,text="Train",bg='blue', fg='white', font=('helvetica',9,'bold'), command=train_eigen_btn_load,width=20, height=1 )
canvas.create_window(170,250, window=trainE_btn)

recogE_btn = Button(root,text="Predict", bg='blue', fg='white', font=('helvetica',9,'bold'), command=recog_eigen_btn_load,width=30, height=4)
canvas.create_window(470,230, window=recogE_btn)

root.mainloop()
