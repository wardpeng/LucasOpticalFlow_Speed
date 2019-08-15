#coding=UTF-8
import numpy as np
import cv2
import time
def filter_sum(list):
    length = len(list)
    if num>0 or num==0:
        return num
    else:
        return 0-num

def ceshi(num):
    t=0
    # cap=cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('2.mov')
    cap = cv2.VideoCapture('2.mov')
    feature_params=dict(maxCorners=100,qualityLevel=0.4,minDistance=7,blockSize=7)
    lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
    color=np.random.randint(0,255,(100,3))
    ret,old=cap.read()
    old_gray=cv2.cvtColor(old,cv2.COLOR_BGR2GRAY)
    p0=cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)
    if num==1:

        l1=[]

    elif num==2:
        while(1):
            try:
                l1=[]
                l2=[]
                d1=0
                d2=0
                ret,frame=cap.read()
                frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                p1,st,err=cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,**lk_params)
                good_new=p1[st==1]
                good_old=p0[st==1]
                #print good_new
                #print good_old
                rows1 = frame.shape[0]
                cols1 = frame.shape[1]
                out = np.zeros((rows1,cols1,3), dtype='uint8')
                out[:rows1,:cols1] = np.dstack([frame_gray, frame_gray, frame_gray])
                lenth=len(good_new)
                for i ,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b=new.ravel()
                    c,d=old.ravel()
                    res1=a-c
                    res2=b-d
                    l1.append(res1)
                    l2.append(res2)
                   # print (int(a),int(b)),(int(c),int(d))
                    cv2.line(out,(int(a),int(b)),(int(c),int(d)),color[i].tolist(),2)
                    cv2.circle(out,(int(a),int(b)),5,color[i].tolist(),-1)
                 #   frame=cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
                d1=sum(l1)/lenth
                d2=sum(l2)/lenth
                cv2.imshow('frame.jpg',out)
                print("X",d1,"Y",d2)
                k=cv2.waitKey(2)&0xff
                if k==27:
                    break
                old_gray=frame_gray.copy()
                p0=good_new.reshape(-1,1,2)
                t=t+1
                if len(good_new)<10:
                    feature_params=dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)
                    lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))
                    color=np.random.randint(0,255,(100,3))
                    ret,old=cap.read()
                    old_gray=cv2.cvtColor(old,cv2.COLOR_BGR2GRAY)
                    p0=cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)
                    t=0
            except:
                time.sleep(0.01)
                print("静止")
    cv2.destroyAllWindows()
    cap.release()
#1为位移检测模式，2为测速模式
ceshi(2)