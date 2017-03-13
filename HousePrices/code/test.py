from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt

#create a array and the -1 index is labels
labels=np.random.rand(100,2)
labels[:60,1]=1
labels[60:,1]=-1
#print ("labels",labels)
score=np.zeros(shape=(100,2))
score[:60,0]=0.7
score[60:,0]=0.3
#print ("score:",score)




def getROC(label,score):
    #TP
    TP=0
    for i in range(100):
        if(label[i,1]==1 and score[i,1]==1):
            TP+=1

    #TN
    TN = 0
    for i in range(100):
        if (label[i, 1] == -1 and score[i, 1] == -1):
            TN += 1

    # FP
    FP = 0
    for i in range(100):
        if (label[i, 1] == -1 and score[i, 1] == 1):
            FP += 1

    # FN
    FN = 0
    for i in range(100):
        if (label[i, 1] == 1 and score[i, 1] == -1):
            FN += 1


    FPR=FP/(FP+TN)
    TPR=TP/(TP+FN)
    return FPR,TPR

def reScore(m,n,score):
    temp=score[:]
    index1 = np.random.random_integers(low=0, high=60, size=(m,))
    index2 = np.random.random_integers(low=60, high=99, size=(n,))

    temp[index1,0]=np.random.rand(m)
    temp[index2, 0] = np.random.rand(n)
    return temp

def reLabel(threshold,score):
    temp=score[:]
    for s in temp:
        if(s[0]>threshold):
            s[1]=1
        else:
            s[1]=-1
    return temp


ms=[50,45,40,35,35,20,20,15,15,10]
ns=[30,30,20,20,15,20,10,15,10,5]



def draw(ms,ns,score):
    ax=plt.subplot(1,1,1)
  #  fig=plt.figure()
   # ax=fig.add_subplot(1,1,1)
    draw_x=[]
    draw_y=[]
    for (m,n) in zip(ms,ns):
        new_score = reScore(m, n, score=score)
        threholds = np.linspace(start=0, stop=1.0, num=1000)
        FPRs = np.zeros(shape=(1000,))
        TPRs = np.zeros(shape=(1000,))
        for i in range(1000):
            new_score = reLabel(threholds[i], new_score)
            FPR_temp, TPR_temp = getROC(labels, new_score)
            FPRs[i] = FPR_temp
            TPRs[i] = TPR_temp
        ax.plot(FPRs,TPRs)

    #    draw_x.append(FPRs)
     #   draw_y.append(TPRs)

    #plt.plot(draw_x,draw_y)
    plt.show()

draw(ms,ns,score)