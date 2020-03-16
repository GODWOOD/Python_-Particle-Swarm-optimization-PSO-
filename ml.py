import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import math
import os 
import time
from decimal import *
import random
import copy

global BestW4D
global BestW6D
global BestE4D
global BestE6D

def changeFloat(number) :
    return float(str(Decimal(str(number)).quantize(Decimal('0.00000000'))))

def changeInt(number) :
    return float(str(Decimal(str(number)).quantize(Decimal('0'))))

def findXY(x1,y1,x2,y2,x,y) :
    ansx=[]
    ansy=[]
    for i in range(0,len(x)-1) :
        try:
            ax = ((x1 - x2) * (x[i] * y[i+1] - x[i+1] * y[i]) - (x[i] - x[i+1]) * (x1 * y2 - x2 * y1))/ ((x[i] - x[i+1]) * (y1 - y2) - (x1 - x2) * (y[i] - y[i+1]))
            ax=changeFloat(ax)
            ay = ((y1 - y2) * (x[i] * y[i+1] - x[i+1] * y[i]) - (x1 * y2 - x2 * y1) * (y[i] - y[i+1]))/ ((y1 - y2) * (x[i] - x[i+1]) - (x1 - x2) * (y[i] - y[i+1]))
            ay=changeFloat(ay)
            if ((ax-x[i])*(ax-x[i+1])<=0 and (ay-y[i])*(ay-y[i+1])<=0) and (((ax-x1)*(ax-x2)<=0) and (ay-y1)*(ay-y2)<=0) :
                correct=True
            else :
                correct=False
        except :
            ax=0
            ay=0
            correct=False
        #print((ax-x1)*(ax-x2),(ay-y1)*(ay-y2))
        if correct :
            ansx.append(ax)
            ansy.append(ay)
    #print("ANS",ansx,ansy)
    #print(x1,y1,x2,y2)
    smalli=-1
    small=100000
    for i in range(0,len(ansx)) :
        a=0
        a=a+math.pow((ansx[i]-x1),2)+math.pow((ansy[i]-y1),2)
        if a<small :
            #print(i,a)
            small=a
            smalli=i
    #print(small,smalli)
    #print("ANS",ansx[smalli],ansy[smalli])
    return ansx[smalli],ansy[smalli]

def toRad(num) :
    return (num*math.pi)/180
    
def checkRad(num):
    if num>=-90*math.pi/180 and num<=1.5*math.pi :
        return num
    else :
        while num<=1.5*math.pi :
            num-=2*math.pi
        while num>=-90*math.pi/180 :
            num+=2*math.pi
        return num

def countDegree(F,R,L,J) :
    global BestW4D
    #print(BestW4D)
    Mstart=J+1
    sum=BestW4D[0]
    for j in range(0,J) :
        a=math.pow(F-BestW4D[Mstart+j*3+0],2)
        a+=math.pow(R-BestW4D[Mstart+j*3+1],2)
        a+=math.pow(L-BestW4D[Mstart+j*3+2],2)
        sum+=math.exp(-1*((a/2)/math.pow(BestW4D[-(j+1)],2)))*BestW4D[j+1]
    if sum>1 :
        sum=1
    elif sum<-1 :
        sum=-1
    #print("DEGREE",sum*40)
    return toRad(sum*40)

def distance(a,b,x,y) :
    return changeFloat(math.pow(math.pow(a-x,2)+math.pow(b-y,2),0.5))

def countSum(F,R,L,J,w4D) :
    Mstart=J+1
    sum=w4D[0]
    for j in range(0,J) :
        a=math.pow(F-w4D[Mstart+j*3+0],2)
        #print(a)
        a+=math.pow(R-w4D[Mstart+j*3+1],2)
        #print(a)
        a+=math.pow(L-w4D[Mstart+j*3+2],2)
        #print("to: ",a)
        sum+=math.exp(-1*((a/2)/math.pow(w4D[-(j+1)],2)))*w4D[j+1]
        #print("*sigma: ",sum)
    
    return sum

def train4D(counter,groups,c1,c2,J) :
    global BestE4D
    global BestW4D
    BestE4D=10
    R=[]
    L=[]
    F=[]
    ans=[]
    loadFile = open("train4dAll.txt",'r')
    Data=loadFile.readline()
    while len(Data)>1 :   #設置數據
        Data=Data.strip().split(" ")
        F.append(float(Data[0])/40-1)
        R.append(float(Data[1])/40-1)
        L.append(float(Data[2])/40-1)
        ans.append(float(Data[3])/40)
        Data.clear()
        Data=loadFile.readline()
    #print(len(ans))
    loadFile.close()
    w=[]
    speed=[]
    for q in range(0,groups) :
        ss=0
        a=[changeFloat(random.uniform(-1,1))]
        b=[changeFloat(random.uniform(-1,1))]
        ss=math.pow(b[-1],2)
        for i in range(0,J*(3+2)) :
            a.append(changeFloat(random.uniform(-1,1)))
            b.append(changeFloat(random.uniform(-1,1)))
            ss=math.pow(b[-1],2)
        w.append(a)
        ss=math.pow(ss,0.5)
        if ss>1 :
            for i in range(0,len(b)) :
                b[i]=b[i]/ss
        elif ss>0.8 :
            for i in range(0,len(b)) :
                b[i]=b[i]*ss
        speed.append(b)
    Mstart=J+1
    E=[0]*len(w)
    Error=[0]*len(w)
    BestE=[1000]*len(w)
    BestW=copy.deepcopy(w)
    check=True
    times=0
    while times<counter and check :#訓練
        #計算F(x) & E(x)
        for i in range(0,len(Error)) :
            Error[i]=0
        #print(E)
        for runtime in range(0,1) :   #一次跌帶要跑幾次
            for qq in range(0,groups) :
                for i in range(0,len(R)) :
                    sum=countSum(F[i],R[i],L[i],J,w[qq])
                        #print("*sigma: ",sum)
                    sum=w[qq][0]
                    for j in range(0,J) :
                        a=math.pow(F[i]-w[qq][Mstart+j*3+0],2)
                        a+=math.pow(R[i]-w[qq][Mstart+j*3+1],2)
                        a+=math.pow(L[i]-w[qq][Mstart+j*3+2],2)
                        sum+=math.exp(-1*((a/2)/math.pow(w[qq][-(j+1)],2)))*w[qq][j+1]
                    if sum>1 :
                        sum=1
                    elif sum<-1 :
                        sum=-1
                    Error[qq]+=abs(ans[i]-sum)/len(ans)
                #print("ZZZ",Error[qq],"XXX")
                if Error[qq]<BestE[qq] :
                    BestE[qq]=Error[qq]
                    BestW[qq]=copy.deepcopy(w[qq])
                if Error[qq]<BestE4D :
                    BestE4D=Error[qq]
                    BestW4D=copy.deepcopy(w[qq])
                    #print("BB",BestE4D)
                    #print(BestW4D)
            #X1+V=X2
            for i in range(0,len(w)) :
                for j in range(0,len(w[i])) :
                    w[i][j]+=speed[i][j]
                    if w[i][j]>1 :
                        w[i][j]=1
                    elif w[i][j]<-1 :
                        w[i][j]=-1
            #print(speed)
            
            for i in range(0,len(speed)) :
                ss=0
                for j in range(0,len(speed[i])) :
                    speed[i][j]=speed[i][j]+(BestW[i][j]-w[i][j])*c1*random.uniform(0,1)+(BestW4D[j]-w[i][j])*c2*random.uniform(0,1)
                    ss+=math.pow(speed[i][j],2)
                ss=math.pow(ss,0.5)
                if ss>1 :
                    for j in range(0,len(speed[i])) :
                        speed[i][j]=speed[i][j]/ss
                elif ss>0.8 :
                    for j in range(0,len(speed[i])) :
                        speed[i][j]=speed[i][j]*ss
            if BestE4D<0.22 :
                #print("AAAAAAAA",BestE4D)
                check=False
        times+=1
        #print(times)
        #print(E)
        #print(Error)
        #print(BestE)
        #print(BestE4D)

        

def test(qqq,counter,groups,c1,c2,J) :
    Tstart=time.time()
    train4D(counter,groups,c1,c2,J)
    Tend=time.time()
    T4D=Tend-Tstart

    global BestW4D
    global BestE4D
    #print("E" ,BestE4D)
    #BestW4D=[-0.06287819517892114, 0.2595325235907913, 0.7959244549555529, -1.0, -0.06405077158407164, 0.0012819486473485348, 0.2480915372531843, 0.026211297544760398, -0.074320724708365, 0.7094709332765032, -1.0, -1.0, -1.0, -0.00655429520457677, 0.00023009100275416967, -1.0, -0.033988195549236246, -0.0717481627124461, 0.5234525574134687, 1.0, -0.8067387063556593]

    filename=qqq
    L4=[]
    L6=[]

    loadFile = open(filename,'r')
    Data=loadFile.readline()
    Data=Data.strip().split(',')
    X=float(Data[0])
    Y=float(Data[1])
    Degree=int(Data[2])
    Degree=checkRad(toRad(Degree))

    Endx=[]
    Endy=[]
    Data=loadFile.readline()
    Data=Data.strip().split(',')
    Endx.append(int(Data[0]))
    Endy.append(int(Data[1]))
    Data=loadFile.readline()
    Data=Data.strip().split(',')
    Endx.append(int(Data[0]))
    Endy.append(int(Data[1]))

    Roadx=[]
    Roady=[]
    Data=loadFile.readline()
    Data=Data.strip().split(',')
    #print(Data)
    while len(Data)>1 :
        Roadx.append(int(Data[0]))
        Roady.append(int(Data[1]))
        Data.clear()
        Data=loadFile.readline()
        Data=Data.strip().split(',')
        #print(Data)
    #print(X,Y,Degree)
    #print(Endx,Endy)
    #print(Roadx,Roady)


    plt.close()  #clf() # 清图  cla() # 清坐标轴 close() # 关窗口
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.axis("equal") #设置图像显示的时候XY轴比例
    plt.grid(True) #添加网格
    plt.ion()  #interactive mode on
    a=[]
    b=[]
    a.append(Roadx[0])
    b.append(Roady[0])
    for i in range(1,len(Roadx)) :
        a.append(Roadx[i])
        b.append(Roady[i])
        plt.plot(a,b,c='b')
        a.remove(a[0])
        b.remove(b[0])
    x=0
    y=0
    r = 3.0
    try:
        while not((Endx[0]-X)*(Endx[1]-X)<=0 and (Endy[0]-Y)*(Endy[1]-Y)<=0):
            ax.scatter(X,Y,c='r',marker='.')  #散点图
            a, b = (X,Y)
            theta = np.arange(0, 2*np.pi, 0.01)
            w = a + r * np.cos(theta)
            q = b + r * np.sin(theta)
            circle=plt.plot(w, q,c='c')
            #print(X,Y)
            #print("RP:")
            rp=findXY(X,Y,X+100*math.cos(Degree+math.pi*(-45)/180),Y+100*math.sin(Degree+math.pi*(-45)/180),Roadx,Roady)
            #print("LP:")
            lp=findXY(X,Y,X+100*math.cos(Degree+math.pi*45/180),Y+100*math.sin(Degree+math.pi*45/180),Roadx,Roady)
            #print("face")
            facep=findXY(X,Y,X+100*math.cos(Degree),Y+100*math.sin(Degree),Roadx,Roady)
            #print("P",lp,";",facep,";",rp)
            ld=distance(X,Y,lp[0],lp[1])
            rd=distance(X,Y,rp[0],rp[1])
            faced=distance(X,Y,facep[0],facep[1])
            #print("Distance",ld,faced,rd)
            newdegree=countDegree((faced-40)/40,(rd-40)/40,(ld-40)/40,J)

            Degree=changeFloat(Degree-np.arcsin(2*math.sin(newdegree)/6))
            Degree=checkRad(Degree)  
            #print("DEGREE",newdegree) 
            X=changeFloat(X+math.cos(newdegree+Degree)+(math.sin(newdegree)*math.sin(Degree)))
            Y=changeFloat(Y+math.sin(newdegree+Degree)-(math.sin(newdegree)*math.cos(Degree)))
            #print(X,Y,Degree)
            #print("")
            #print("")

            L4.append(str(changeFloat(faced)))
            L4.append(str(changeFloat(rd)))
            L4.append(str(changeFloat(ld)))
            L4.append(str(changeFloat(newdegree*180/math.pi)))

            L6.append(str(changeFloat(X)))
            L6.append(str(changeFloat(Y)))
            L6.append(str(changeFloat(faced)))
            L6.append(str(changeFloat(rd)))
            L6.append(str(changeFloat(ld)))
            L6.append(str(changeFloat(newdegree*180/math.pi)))

            plt.pause(0.1)
            #os.system("pause")
            ax.lines.remove(circle[0])

    except Exception as err:
        print(err)
    #print("A")
    f4 = open('train4D.txt', 'w')
    f6 = open('train6D.txt', 'w')
    #print(L4)
    #print(L6)
    S4=""
    S6=""
    for i in range(0,len(L4)) :
        S4=S4+L4[i]+" "
        if (i-3)%4 == 0 :
            S4=S4+"\n" 
    for i in range(0,len(L6)) :
        S6=S6+L6[i]+" "
        if (i-5)%6 == 0 :
            S6=S6+"\n" 

    f4.write(S4)
    f6.write(S6)       
    f4.close()
    f6.close()
    print(T4D)
    #print(BestW4D)

    #os.system("pause")