# coding: utf-8
# YuanYang
import math
import cv2
import numpy as np


def nms(boxes, overlap_threshold, mode='Union'):
    """
        non max suppression

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # 依次取出左上角和右下角坐标以及分类器得分(置信度)
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)    #求取每个bbox的面积
    idxs = np.argsort(score)    #对bbox的score按从小到大的顺序排序得到idxs

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # 每次都从idxs的末尾开始取值，并将index放到pick列表中
        last = len(idxs) - 1    # 当前剩余框的数量
        i = idxs[last]  # 选中最后一个，即得分最高的框
        pick.append(i)

        # 计算两个框的交集的左上角坐标（xx1，yy1）和右下角坐标（xx2，yy2），不管有无交集，都可以得到这4个值。
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        # 将计算bbox的宽度和高度，如果宽度或高度是负值（也就是说不存在这样的bbox，再往前追溯，就是两个框没有交集，
        # 因此生成的左上角坐标（xx1，yy1）和右下角（xx2，yy2）构成不了一个框），这样的话就用0值代替
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        # 不同定义下的IOU
        if mode == 'Min':       #mode默认采用’Union’
            # 重叠面积与最小框面积的比值
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            # 交集面积/并集面积
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # 将idxs中overlap满足阈值的bbox的index删除
        good_index = np.where(overlap > overlap_threshold)[0]   #返回满足这个条件表达式的bbox的index
        com_index = np.concatenate(([last],  good_index)) # concatenate操作就是将原来socre最大的那个bbox的index和现在满足条件的bbox的index合并成一个numpy array
        #np.delete函数从idxs中删掉指定bbox的index
        idxs = np.delete(idxs, com_index)

    return pick

def adjust_input(in_data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

    out_data = out_data.transpose((2,0,1))
    out_data = np.expand_dims(out_data, 0)
    # 对图像进行预处理（中心化）
    out_data = (out_data - 127.5)*0.0078125
    return out_data

def generate_bbox(map, reg, scale, threshold):
     """
         generate bbox from feature map
     Parameters:
     ----------
         map: numpy array , n x m x 1
             detect score for each position
         reg: numpy array , n x m x 4
             bbox
         scale: float number
             scale of this detection
         threshold: float number
             detect threshold
     Returns:
     -------
         bbox array
     """
     stride = 2
     cellsize = 12

     t_index = np.where(map>threshold)

     # find nothing
     if t_index[0].size == 0:
         return np.array([])

     dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

     reg = np.array([dx1, dy1, dx2, dy2])
     score = map[t_index[0], t_index[1]]
     boundingbox = np.vstack([np.round((stride*t_index[1]+1)/scale),
                              np.round((stride*t_index[0]+1)/scale),
                              np.round((stride*t_index[1]+1+cellsize)/scale),
                              np.round((stride*t_index[0]+1+cellsize)/scale),
                              score,
                              reg])

     return boundingbox.T


def detect_first_stage(img, net, scale, threshold):
    """
        run PNet for first stage
    
    Parameters:
    ----------
        img: numpy array, bgr order
            input image
        scale: float number
            how much should the input image scale
        net: PNet
            worker
    Returns:
    -------
        total_boxes : bboxes
    """
    height, width, _ = img.shape
    hs = int(math.ceil(height * scale))
    ws = int(math.ceil(width * scale))
    
    im_data = cv2.resize(img, (ws,hs))
    
    # adjust for the network input
    input_buf = adjust_input(im_data)
    output = net.predict(input_buf)

    # 考虑到PNet的输入实在太小，因此在训练的时候很难截取到完全合适的人脸，因此训练边界框的生成时广泛采用了部分样本。
    # 因此，PNet直接输出的边界框并不是传统回归中的边界坐标，而是预测人脸位置相对于输入图片的位置差。
    # 所以，需要专门的算法将位置差转换为真实位置。
    boxes = generate_bbox(output[1][0,1,:,:], output[0], scale, threshold)

    if boxes.size == 0:
        return None

    # nms
    pick = nms(boxes[:,0:5], 0.5, mode='Union')
    boxes = boxes[pick]
    return boxes

def detect_first_stage_warpper( args ):
    return detect_first_stage(*args)
