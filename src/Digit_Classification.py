import cv2
import numpy as np
import glob



def networkinference(w, img):
    size = np.shape(img)
    img = np.reshape(img, (size[0]*size[1],1))
    layer1 = w.dot(img)
    layer2 = softmax(layer1)
    return layer2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def loss_fuction(y, groundtruth):
    y = np.log(y)
    s = np.multiply(y, groundtruth)
    return -np.sum(s)

def gradient_decent(step, w0, groundtruth, img):
    #print np.shape(w)
    size = np.shape(img)
    img = np.reshape(img, (size[0]*size[1],1))
    layer1 = w0.dot(img)
    layer2 = softmax(layer1)
    j1 = np.concatenate((-groundtruth[0]/layer2[0],-groundtruth[1]/layer2[1]),axis=1)
    j21 = np.concatenate((layer2[0]*(1-layer2[0]),-layer2[0]*layer2[1]),axis = 1)
    j22 = np.concatenate((-layer2[1]*layer2[0],layer2[1]*(1-layer2[1])),axis = 1)
    j2 = np.array([j21,j22])
    zero = np.zeros((1,np.size(img)))
    j31 = np.concatenate((np.transpose(img),zero),axis = 1)
    j32 = np.concatenate((zero,np.transpose(img)),axis = 1)
    j3 = np.concatenate((j31,j32), axis = 0)
    J = j1.dot(j2)
    J = J.dot(j3)
    loss = loss_fuction(layer2, groundtruth)
    J = np.transpose(np.array([J]))
    w0 = np.reshape(w0, (1, 1568))
    w0 = np.transpose(w0)
    w1 = w0 - step*J.dot(loss)
    w1 = np.transpose(w1)
    w1 = np.hsplit(w1, 2)
    w1 = np.concatenate((w1[0],w1[1]),axis = 0)
    return w1
    
    
    
      


if __name__ == '__main__':
    w = np.ones((2,28*28))
    w = w/1000
    minw = w
    first = True
    groundtruth = np.array([[1],[0]])
    step = 0.00000001
    itr = 1000
    for n in glob.glob("train/0/*.png"):
        img = cv2.imread(n,0)
        while itr>=0 :
            if first:
                first = False
                layer2 = networkinference(w, img)
                minloss = loss_fuction(layer2, groundtruth)
                w = gradient_decent(step, w, groundtruth, img)
            else:
                w = gradient_decent(step, w, groundtruth, img)
                layer2 = networkinference(w, img)
                loss = loss_fuction(layer2, groundtruth)
                if minloss>loss:
                    minloss = loss
                    minw = w
                else:
                    break
            itr = itr-1
    first = True
    itr = 1000
    groundtruth = np.array([[0.2],[0.8]])
    for n in glob.glob("train/1/*.png"):
        img = cv2.imread(n,0)
        while itr>=0 :
            if first:
                first = False
                layer2 = networkinference(w, img)
                minloss = loss_fuction(layer2, groundtruth)
                w = gradient_decent(step, w, groundtruth, img)
            else:
                w = gradient_decent(step, w, groundtruth, img)
                layer2 = networkinference(w, img)
                loss = loss_fuction(layer2, groundtruth)
                if minloss>loss:
                    minloss = loss
                    minw = w
                else:
                    break
            itr = itr-1
    layer2 = []
    sort1 = []
    sort2 = []
    count = 0
    for n in glob.glob("test/0/*.png"):
        img = cv2.imread(n,0)
        layer2 = networkinference(minw, img)
        count = count+1
        #print layer2
        if layer2[0]>layer2[1] :
            sort1 = np.append(sort1, [1], axis=1)
        else:
            sort1 = np.append(sort1, [0], axis=1)
    err1 = (count-np.sum(sort1))/count
    print 'error of detect 0: ' 
    print err1
    count = 0
    for n in glob.glob("test/1/*.png"):
        img = cv2.imread(n,0)
        layer2 = networkinference(minw, img)
        count = count+1
        #print layer2
        if layer2[0]<layer2[1] :
            sort2 = np.append(sort2, [1], axis=1)
        else:
            sort2 = np.append(sort2, [0], axis=1)
    err2 = (count-np.sum(sort2))/count
    print 'error of detect 1: '
    print err2
    print sort1
    print sort2



        