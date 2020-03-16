import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import ml
import random
import sys


def clickstart() :
    FileName=entry_0.get()
    entry_0.delete(0,'end')
    #FileName="test"
    #print(FileName)
    try:
        counter=int(entry_1.get())
        entry_1.delete(0,'end')
    except:
        counter=20
    try:
        groups=int(entry_2.get())
        entry_2.delete(0,'end')
    except:
        groups=10
    try:
        c1=float(entry_3.get())
        mutation=mutation*10
        entry_3.delete(0,'end')
    except:
        c1=0.2
    try:
        c2=float(entry_4.get())
        mating=mating*10
        entry_4.delete(0,'end')
    except:
        c2=0.5
    try:
        J=int(entry_5.get())
        entry_5.delete(0,'end')
    except:
        J=4
    ml.test(FileName,counter,groups,c1,c2,J)
#-----------create GUI---------
FileName=""
face=tk.Tk()
face.title("DaJaVu")

#-------label+entry(row=0)-----------
label_0=tk.Label(face,text="資料檔名:")
label_0.grid(column=0,row=0)

entry_0=tk.Entry(face)
entry_0.grid(column=1,row=0)

label_1=tk.Label(face,text="迭代次數:")
label_1.grid(column=0,row=1)

entry_1=tk.Entry(face)
entry_1.grid(column=1,row=1)

label_2=tk.Label(face,text="族群大小:")
label_2.grid(column=0,row=2)

entry_2=tk.Entry(face)
entry_2.grid(column=1,row=2)

label_3=tk.Label(face,text="c1:")
label_3.grid(column=0,row=3)

entry_3=tk.Entry(face)
entry_3.grid(column=1,row=3)

label_4=tk.Label(face,text="c2:")
label_4.grid(column=0,row=4)

entry_4=tk.Entry(face)
entry_4.grid(column=1,row=4)

label_5=tk.Label(face,text="網路J值:")
label_5.grid(column=0,row=5)

entry_5=tk.Entry(face)
entry_5.grid(column=1,row=5)

button=tk.Button(face,text="start",command=clickstart)
button.grid(column=0,row=6)

#----run GUI---------
face.mainloop()

