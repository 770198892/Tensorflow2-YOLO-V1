import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from YOLONetModel import *
import config as cfg
from tensorflow import keras
import tensorflow.compat.v1 as tf1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
boundary1=cfg.CELL_SIZE*cfg.CELL_SIZE*len(cfg.CLASSES)
boundary2=boundary1+cfg.CELL_SIZE*cfg.CELL_SIZE*cfg.BOXES_PER_CELL

def detect(img):
    img_h,img_w,_=img.shape
    inputs=cv2.resize(img,(cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)).astype(np.float32)
    inputs=(inputs/255.0-0.5)*2
    inputs=np.reshape(inputs,(1,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,3))
    result=detect_from_cvmat(inputs)[0]
    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / cfg.IMAGE_SIZE)
        result[i][2] *= (1.0 * img_h / cfg.IMAGE_SIZE)
        result[i][3] *= (1.0 * img_w / cfg.IMAGE_SIZE)
        result[i][4] *= (1.0 * img_h / cfg.IMAGE_SIZE)
    return result
def detect_from_cvmat(inputs):
    output= YOLO_net.predict(inputs)
    results=[]
    for i in range(output.shape[0]):
        results.append(interpret_output(output[i]))
    return results


def interpret_output(output):
    # 有目标情况下的分类概率
    probs = np.zeros((cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, len(cfg.CLASSES)))
    # 分类概率
    class_probs = np.reshape(output[0:boundary1], (cfg.CELL_SIZE, cfg.CELL_SIZE, len(cfg.CLASSES)))
    # boxes有目标概率
    scales = np.reshape(output[boundary1:boundary2], (cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL))
    # boxes及偏移量修正
    boxes = np.reshape(output[boundary2:], (cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 4))
    offset = np.transpose(np.reshape(np.array([np.arange(cfg.CELL_SIZE)] *
                                              cfg.CELL_SIZE * cfg.BOXES_PER_CELL),
                                     [cfg.BOXES_PER_CELL, cfg.CELL_SIZE,cfg.CELL_SIZE]), (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, :2] / cfg.CELL_SIZE
    boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
    boxes *= cfg.IMAGE_SIZE

    for i in range(cfg.BOXES_PER_CELL):
        for j in range(len(cfg.CLASSES)):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])
    # 保留分类概率大于阈值的 shape=[7,7,2,20]
    filter_mat_probs = np.array(probs >= cfg.THRESHOLD, dtype='bool')
    # 保留的下标,长度为4的元组对应ceil,ceil,box_per_cell,class_prob，每个元组长度为保留boxes的个数
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    # 保留boxes的坐标
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    # 保留boxes的置信度
    probs_filtered = probs[filter_mat_probs]
    # 选框目标类别编号
    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
        filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    # 选框概率从大到小的下标
    argsort = np.array(np.argsort(probs_filtered))[::-1]
    # 概率由大到小boxes坐标
    boxes_filtered = boxes_filtered[argsort]
    # 概率由大到小boxes置信度
    probs_filtered = probs_filtered[argsort]
    # 概率由大到小类别
    classes_num_filtered = classes_num_filtered[argsort]

    # 非极大值抑制，将iou大的删掉
    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > cfg.IOU_THRESHOLD:
                probs_filtered[j] = 0.0
    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    # 非极大值抑制后的参数
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]
    result = []
    for i in range(len(boxes_filtered)):
        result.append([cfg.CLASSES[classes_num_filtered[i]], boxes_filtered[i][0],
                       boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])
    return result

def iou(box1,box2):
    #重叠部分长高
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
def draw_result(img,result):
    for i in range(len(result)):
        x=int(result[i][1])
        y=int(result[i][2])
        w=int(result[i][3]/2)
        h=int(result[i][4]/2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
def load_weights(model,file_path):

    keys = [('yolo/conv_2/biases', 'yolo/conv_2/weights'),('yolo/conv_4/biases', 'yolo/conv_4/weights'),
            ('yolo/conv_6/biases', 'yolo/conv_6/weights'),('yolo/conv_7/biases' , 'yolo/conv_7/weights'),
            ('yolo/conv_8/biases' , 'yolo/conv_8/weights'), ('yolo/conv_9/biases' , 'yolo/conv_9/weights'),
            ('yolo/conv_11/biases', 'yolo/conv_11/weights'),('yolo/conv_12/biases', 'yolo/conv_12/weights'),
            ('yolo/conv_13/biases', 'yolo/conv_13/weights'),('yolo/conv_14/biases', 'yolo/conv_14/weights'),
            ('yolo/conv_15/biases', 'yolo/conv_15/weights'),('yolo/conv_16/biases', 'yolo/conv_16/weights'),
            ('yolo/conv_17/biases', 'yolo/conv_17/weights'),('yolo/conv_18/biases', 'yolo/conv_18/weights'),
            ('yolo/conv_19/biases', 'yolo/conv_19/weights'),('yolo/conv_20/biases','yolo/conv_20/weights'),
            ('yolo/conv_22/biases', 'yolo/conv_22/weights'),('yolo/conv_23/biases', 'yolo/conv_23/weights'),
            ('yolo/conv_24/biases', 'yolo/conv_24/weights'),('yolo/conv_25/biases','yolo/conv_25/weights'),
            ('yolo/conv_26/biases', 'yolo/conv_26/weights'),('yolo/conv_28/biases','yolo/conv_28/weights'),
            ('yolo/conv_29/biases', 'yolo/conv_29/weights'),('yolo/conv_30/biases','yolo/conv_30/weights'),
            ('yolo/fc_33/biases', 'yolo/fc_33/weights'),('yolo/fc_34/biases', 'yolo/fc_34/weights'),
            ('yolo/fc_36/biases', 'yolo/fc_36/weights')]
    reader = tf1.train.NewCheckpointReader(file_path)
    step = 0
    for idx,key in enumerate(keys):
        bias = reader.get_tensor(key[0])
        weight = reader.get_tensor(key[1])
        w = [weight, bias]
        tt = model.layers[idx+step].get_weights()
        if len(tt) == 0:
            step = step + 1
        print(idx + step)
        model.layers[idx+step].set_weights(w)
# 重构模型
YOLO_net = YOLONetModel()
YOLO_net.load_weights("./YOLO.ckpt")
# images = plt.imread(cfg.TEST_PATH+'2007_000464.jpg')
# result = detect(images)
# load_weights(YOLO_net,"./YOLO_small.ckpt")
for filename in os.listdir(cfg.TEST_PATH):
    images = plt.imread(cfg.TEST_PATH+filename)
    result = detect(images)
    draw_result(images,result)
    plt.imshow(images)
    plt.show()
#YOLO_net.save_weights("./YOLO.ckpt")
