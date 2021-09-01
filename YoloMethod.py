import tensorflow as tf
from tensorflow import keras
from pascal_voc import *
import numpy as np
import config as cfg
from YOLONetModel import *
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# 按需分配显存
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
def calIOU(boxs_1,boxs_2):
    # 获取左上角 右下角
    boxs_1 = tf.stack([boxs_1[:, :, :, :, 0] - boxs_1[:, :, :, :, 2] / 2.0,
                       boxs_1[:, :, :, :, 1] - boxs_1[:, :, :, :, 3] / 2.0,
                       boxs_1[:, :, :, :, 0] + boxs_1[:, :, :, :, 2] / 2.0,
                       boxs_1[:, :, :, :, 1] + boxs_1[:, :, :, :, 3] / 2.0,
                       ],axis=-1)
    boxs_2 = tf.stack([boxs_2[:, :, :, :, 0] - boxs_2[:, :, :, :, 2] / 2.0,
                       boxs_2[:, :, :, :, 1] - boxs_2[:, :, :, :, 3] / 2.0,
                       boxs_2[:, :, :, :, 0] + boxs_2[:, :, :, :, 2] / 2.0,
                       boxs_2[:, :, :, :, 1] + boxs_2[:, :, :, :, 3] / 2.0,
                       ], axis=-1)
    # 求交叉部分
    a = tf.maximum(boxs_1[:,:,:,:,:2],boxs_2[:,:,:,:,:2])
    b = tf.minimum(boxs_1[:,:,:,:,2:],boxs_2[:,:,:,:,2:])

    # 坐标差
    c = tf.maximum(0.0,b-a)
    # 交叉面积
    Crossarea = c[:,:,:,:,0]*c[:,:,:,:,1]
    # 总面积
    Totalarea = (boxs_1[:,:,:,:,2] - boxs_1[:,:,:,:,0]) * (boxs_1[:,:,:,:,3] - boxs_1[:,:,:,:,1]) + (boxs_2[:,:,:,:,2] - boxs_2[:,:,:,:,0]) * (boxs_2[:,:,:,:,3] - boxs_2[:,:,:,:,1])

    iou = tf.clip_by_value(Crossarea/Totalarea, 0.0, 1.0)
    return iou
# 损失函数 1
def yolo_loss(model,x,y):
    ### 数据格式 ：[None,7,7,25]
    y = tf.constant(y.tolist(),tf.float32)
    y_ = model(x)
    # y_ = tf.reshape(y_,[cfg.BATCH_SIZE,cfg.CELL_SIZE,cfg.CELL_SIZE,len(cfg.CLASSES)+cfg.BOXES_PER_CELL*5])
    ## 未进行归一化
    # predict_class = y_[:,:,:,:20]
    # predict_scale = y_[:,:,:,20:22]
    # predict_boxs  = y_[:,:,:,22:]
    predict_class = tf.reshape(y_[:,:7*7*20],[cfg.BATCH_SIZE,cfg.CELL_SIZE,cfg.CELL_SIZE,len(cfg.CLASSES)])
    predict_scale = tf.reshape(y_[:,7*7*20:7*7*22],[cfg.BATCH_SIZE,cfg.CELL_SIZE,cfg.CELL_SIZE,cfg.BOXES_PER_CELL])
    predict_boxs  = tf.reshape(y_[:,7*7*22:],[cfg.BATCH_SIZE,cfg.CELL_SIZE,cfg.CELL_SIZE,cfg.BOXES_PER_CELL,4])
    #predict_boxs = tf.reshape(predict_boxs,[cfg.BATCH_SIZE,7,7,2,4])
    true_scale = tf.reshape(y[:,:,:,0],[cfg.BATCH_SIZE,7,7,1])
    true_boxs = tf.divide(y[:,:,:,1:5],448.0) # 归一化
    true_boxs = tf.reshape(true_boxs,[cfg.BATCH_SIZE,7,7,1,4])
    true_boxs = tf.tile(true_boxs,[1,1,1,2,1])
    true_class = y[:,:,:,5:]

    # 位置偏移量
    offset = np.transpose(np.reshape(np.array([np.arange(7)] * 7 * 2),
                                     (2, 7, 7)), (1, 2, 0))
    offset = tf.constant(offset,tf.float32)
    offset = tf.reshape(offset, [1, 7, 7, 2])
    offset = tf.tile(offset, [cfg.BATCH_SIZE, 1, 1, 1])
    predict_boxs = tf.stack([(predict_boxs[:,:,:,:,0]+offset)/7,
                                (predict_boxs[:,:,:,:,1]+tf.transpose(offset,(0,2,1,3)))/7,
                                tf.square(predict_boxs[:,:,:,:,2]), # 求平方
                                tf.square(predict_boxs[:,:,:,:,3])],axis=-1)

    # 计算IOU
    iou_predict_truth = calIOU(predict_boxs,true_boxs)
    true_boxs = tf.stack([true_boxs[:,:,:,:,0]*7-offset,
                                true_boxs[:,:,:,:,1]*7-tf.transpose(offset,(0,2,1,3)),
                                tf.sqrt(true_boxs[:,:,:,:,2]), # 求平方根
                                tf.sqrt(true_boxs[:,:,:,:,3])],axis=-1)

    object_mask = tf.reduce_max(iou_predict_truth,3,keepdims=True)
    object_mask = tf.cast((iou_predict_truth>=object_mask),dtype=tf.float32) * true_scale

    noobject_mask = tf.ones_like(object_mask,dtype=tf.float32) - object_mask


    # 计算类别loss
    loss_class = tf.square(tf.multiply(true_scale,tf.subtract(true_class,predict_class)))
    loss_class = tf.reduce_sum(loss_class,axis=[1,2,3])
    loss_class = tf.reduce_mean(loss_class)
    # 计算预测框的loss
    coord_mask = tf.expand_dims(object_mask, 4)
    boxs_loss = coord_mask * (predict_boxs - true_boxs)
    boxs_loss = tf.reduce_sum(tf.square(boxs_loss),axis=[1,2,3,4])
    boxs_loss = tf.reduce_mean(boxs_loss)
    # 计算有目标置信度
    object_scale = object_mask * (predict_scale - iou_predict_truth)
    object_scale = tf.reduce_sum(tf.square(object_scale),axis=[1,2,3])
    object_scale = tf.reduce_mean(object_scale)
    # 计算无目标置信度
    noobject_scale = noobject_mask * predict_scale
    noobject_scale = tf.reduce_sum(tf.square(noobject_scale),axis=[1,2,3])
    noobject_scale = tf.reduce_mean(noobject_scale)
    coord = 5
    noobj = 0.5
    loss = coord * boxs_loss + noobj * noobject_scale + loss_class + object_scale
    return loss
# 计算梯度
def grad(model,inputs,targets):
    with tf.GradientTape() as tape:
        loss_values = yolo_loss(model, inputs, targets)
    return loss_values,tape.gradient(loss_values,model.trainable_variables)

yolo_net = keras.models.Sequential([
    keras.layers.Conv2D(64,7,strides=2,input_shape=[448,448,3],padding='SAME',activation='relu'),
    keras.layers.MaxPool2D(2,2,padding='SAME'),
    keras.layers.Conv2D(192,3,padding='SAME',activation='relu'),
    keras.layers.MaxPool2D(2,2,padding='SAME'),
    keras.layers.Conv2D(128,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(256,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(256,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(512,3,padding='SAME',activation='relu'),
    keras.layers.MaxPool2D(2,2,padding='SAME'),
    keras.layers.Conv2D(256,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(512,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(256,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(512,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(256,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(512,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(256,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(512,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(512,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(1024,3,padding='SAME',activation='relu'),
    keras.layers.MaxPool2D(2,2,padding='SAME'),
    keras.layers.Conv2D(512,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(1024,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(512,1,padding='SAME',activation='relu'),
    keras.layers.Conv2D(1024,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(1024,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(1024,3,2,padding='SAME',activation='relu'),
    keras.layers.Conv2D(1024,3,padding='SAME',activation='relu'),
    keras.layers.Conv2D(1024,3,padding='SAME',activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(4096,activation='relu'),
    keras.layers.Dense(1470,activation='relu'),
    keras.layers.Reshape([7,7,30])
])
#model.build(input_shape=[448,448,3])
pascal = pascal_voc('train')

# 设置优化器
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
train_loss_results = []
train_accuracy_results = []
model = YOLONetModel()
ckpt = tf.train.Checkpoint(net=model)
manager = tf.train.CheckpointManager(ckpt,'./tf_ckpts',max_to_keep=3)
#ckpt.restore("./YOLO_small.ckpt")
epoch_loss_avg = tf.keras.metrics.Mean()
for epoch in range(10000):
    images, labels = pascal.get()
    epoch_loss_avg.reset_states()
    # 优化模型
    loss_value, grads = grad(model,images,labels)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    # 追踪进度
    epoch_loss_avg(loss_value)
    train_loss_results.append(epoch_loss_avg.result())
    print("Epoch {:03d}: Loss: {:.3f}".format(epoch,loss_value))
    if epoch%100 == 0:
        save_path = manager.save()
        print(save_path)

tf.saved_model.save(model,"./model")