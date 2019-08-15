# coding=UTF-8
import numpy as np
import cv2
import time
import scipy.signal as signal
import pylab as pl
import math
import traceback

# def filter_sum(list):
#     length = len(list)
#     if num>0 or num==0:
#         return num
#     else:
#         return 0-num

def ArithmeticAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean


'''
MedianAverage
'''


def MedianAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp, np.where(tmp == tmp.max())[0], axis=0)
        tmp = np.delete(tmp, np.where(tmp == tmp.min())[0], axis=0)
        mean.append(tmp.mean())
    return mean


def filterTest():
    T = np.arange(0, 0.5, 1 / 4410.0)
    num = signal.chirp(T, f0=10, t1=0.5, f1=1000.0)
    pl.subplot(2, 1, 1)
    pl.plot(num)
    result = MedianAverage(num.copy(), 30)

    # print(num - result)
    pl.subplot(2, 1, 2)
    pl.plot(result)
    pl.show()


def GetSpeed():
    t = 0
    speedResult=[]
    bRunning = True
    # cap=cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('2.mov')
    cap = cv2.VideoCapture('../Source/20190810-2-2.mov')
    feature_params = dict(maxCorners=100, qualityLevel=0.4, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (100, 3))
    ret, old = cap.read()
    old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    while (bRunning):
        try:
            l1 = []
            l2 = []
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # print good_new
            # print good_old
            rows1 = frame.shape[0]
            cols1 = frame.shape[1]
            out = np.zeros((rows1, cols1, 3), dtype='uint8')
            out[:rows1, :cols1] = np.dstack([frame_gray, frame_gray, frame_gray])
            lenth = len(good_new)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                res1 = a - c
                res2 = b - d
                l1.append(res1)
                l2.append(res2)
                # print (int(a),int(b)),(int(c),int(d))
                cv2.line(out, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                cv2.circle(out, (int(a), int(b)), 5, color[i].tolist(), -1)
            #   frame=cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
            # d1=sum(l1)/lenth
            # d2=sum(l2)/lenth
            d1 = averageWithFilter(l1)
            d2 = averageWithFilter(l2)

            # cv2.imshow('frame.jpg',out)
            # print("X", d1, "Y", d2)
            sum = math.pow(d1,2)+math.pow(d2,2)
            if sum > 0:
                # print("power,sqrt",np.sqrt(sum))
                result_sqrt = np.sqrt(sum)
                print(result_sqrt)
                speedResult.append(result_sqrt)
            k = cv2.waitKey(2) & 0xff
            if k == 27:
                break
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            t = t + 1
            if len(good_new) < 10:
                feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                lk_params = dict(winSize=(15, 15), maxLevel=2,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                color = np.random.randint(0, 255, (100, 3))
                ret, old = cap.read()
                old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                t = 0
        except Exception as e:
            myPlot(speedResult)
            time.sleep(0.01)
            bRunning = False
            print("except######:" )
            print(e)
            cv2.destroyAllWindows()
            cap.release()

def myPlot(data):

    newData = []
    print("before:", len(data))
    print(data)
    array = np.array(data)
    for i in range(0,10):
        array = np.delete(array, np.where(array == array.max())[0], axis=0)
        array = np.delete(array, np.where(array == array.min())[0], axis=0)
    pl.plot(array.tolist())
    print("after:", len(array))
    pl.show()


    # for i in data:
    #     if i < 40:
    #         newData.append(i)
    # print("after:", len(newData))
    # pl.plot(newData)
    # pl.show()

def myTest():
    list = []
    list.append(1.1)
    list.append(1.2)
    list.append(1.3)
    list.append(1.4)
    list.append(1.5)
    # list.append(100.0)
    # list.append(100.0)
    # list.append(-2000.0)
    # list.append(-2000.0)
    # averageWithFilter(list)

    # print(averageWithFilter(list))
    # print(np.array(list).mean())
    pl.plot(list)
    pl.show()

def averageWithFilter(list):
    length = len(list)
    array = np.array(list)
    # All 0
    if len(array)<=4 or len(np.where(array == array.max())[0]) > 5:
        return 0
    # print("===========")
    # print(array)

    # 1/2 valid date: delete min 1/4, max 1/4
    for num in range(0,length,3):
        # print("num" + str(num))
        if len(array) > 1:
            array = np.delete(array, np.where(array == array.max())[0], axis=0)
        if len(array)>0:
            array = np.delete(array, np.where(array == array.min())[0], axis=0)
    # print(array)
    # print(array.mean())
    return array.mean()


if __name__ == '__main__':
    GetSpeed()
    # filterTest()
    # myTest()
