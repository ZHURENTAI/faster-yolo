# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import datetime
import cv2
import matplotlib.pyplot as plt
import sys
#caffe_root = '/home/ttiger/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0,  '../python')
import caffe
import math
mean = np.require([104, 117, 123], dtype=np.float32)[:, np.newaxis, np.newaxis]
caffe.set_mode_cpu()
model_def = 'shuffle2_cut_deploy_intel.prototxt'
#model_weights = 'models/shuffle2_iter_50000.caffemodel'
model_weights = 'models/shuffle2_finetune_iter_60000.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)     



def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))


def overlap(x1, w1, x2, w2): #x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes,key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue
        
        truth =  sorted_boxes[i]
        for j in range(i+1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1
    
    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res

def det(res, im,side):	
	classes = 20
        num = 5
	pred = classes + 4 + 1
        swap = np.zeros((side * side, num, pred))
	index = 0
	'''for h in range(side):
    		for w in range(side):
        		for c in range(pred * num):
            			swap[h * side + w][c / (pred)][c % (pred)]  = res[c][h][w]'''
	biases = [1.32, 1.73, 3.19, 4.01, 5.06, 8.10, 9.47, 4.84, 11.24, 10.01]
	boxes = list()
	for h in range(side):
    		for w in range(side):
                        for c in range(pred * num):
            			swap[h * side + w][c / (pred)][c % (pred)]  = res[c][h][w]
        		for n in range(num):
            			box = list();
            			cls = list();
            			s = 0;
            			x = (w + sigmoid(swap[h * side + w][n][0])) * 1.0 / side;#res[]
            			y = (h + sigmoid(swap[h * side + w][n][1])) * 1.0 / side;
            			ww = (math.exp(swap[h * side + w][n][2]) * biases[2*n]) * 1.0 / side;
            			hh = (math.exp(swap[h * side + w][n][3])*biases[2*n+1]) * 1.0 / side;
            			obj_score = sigmoid(swap[h * side + w][n][4]);
            			for p in range(classes):
                			cls.append(swap[h * side + w][n][5 + p]);
            
            			large = max(cls);
            			for i in range(len(cls)):
                			cls[i] = math.exp(cls[i] - large);
            
            			s = sum(cls);
            			for i in range(len(cls)):
                			cls[i] = cls[i] * 1.0 / s;
            			box.append(x);
            			box.append(y);
            			box.append(ww);
            			box.append(hh);
            			box.append(cls.index(max(cls))+1)
            			box.append(obj_score);
            			box.append(max(cls));
				box.append(obj_score * max(cls))
                                '''if box[4]==10 and box[7] > 0.3:
                			boxes.append(box);
                                if box[4]==7 and box[7] > 0.3:
                			boxes.append(box);
                                if box[4]==9 and box[7] > 0.15:
                			boxes.append(box);
            			if box[4]==15 and box[7] > 0.4:
                			boxes.append(box);
            			if box[4]==20 and box[7] > 0.15:
                			boxes.append(box);'''
            			if box[7] > 0.25:
                			boxes.append(box);


	resbox = apply_nms(boxes, 0.15)
	label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}
	w = im.shape[1]
	h = im.shape[0]
	for box in resbox:
		xmin = (box[0]-box[2]/2.0) * w;
		xmax = (box[0]+box[2]/2.0) * w;
		ymin = (box[1]-box[3]/2.0) * h;
		ymax = (box[1]+box[3]/2.0) * h;
	        if xmin < 0:
			xmin = 0
		if xmax > w:
			xmax = w
		if ymin < 0:
			ymin = 0
		if ymax > h:
			ymax = h
		ids = label_name[box[4]]	
    		display_txt = '{}, {:0.2f} '.format(label_name[box[4]], box[5])
        	cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax),int(ymax)), (12*box[4],255-12*box[4],130+12*(box[4]-10)), 2)
                im = cv2.putText(im, display_txt, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        return  im

cap=cv2.VideoCapture(0)
while True:
  begin = datetime.datetime.now()
  ret,frame=cap.read();
  img = cv2.resize(frame , (448, 448))
  img = np.require(img.transpose((2, 0, 1)), dtype=np.float32)
  img -= mean
  net.blobs['data'].data[...] = img
  output = net.forward()
  res = output['conv_reg'][0]  
  frame=det(res,frame,14)
  cv2.imshow("test",frame)
  cv2.waitKey(3)
  end = datetime.datetime.now()
  k = end - begin
  print(1/k.total_seconds())
