import torch 
import pickle as pkl 
import random
import cv2
import numpy as np


def non_max_sup(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = torch.sort(scores, descending = True)

    keep = []
    while len(order.shape) > 0 and order.shape[0] > 0:
        i = order[0]
        keep.append(i) 
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1 + 1, min=0.0)
        h = torch.clamp(yy2 - yy1 + 1, min=0.0)
        inter = w * h
  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = torch.nonzero(ovr <= thresh).squeeze()
        order = order[inds + 1]

    return keep


def predict_transform(prediction, inp_dim, anchors, num_classes):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2) 
    grid_size = inp_dim // stride 

    bbox_attrs = 5 + num_classes 
    num_anchors = len(anchors)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if prediction.is_cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if prediction.is_cuda:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Softmax the class scores 
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    prediction[:,:,:4] *= stride
    
    return prediction


def sift_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    box_a = prediction.new(prediction.shape)
    #print("boxa_shape", box_a.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    
    # 将中点-宽高转化为左上-右下坐标
    
    batch_size = prediction.size(0)

    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        prob, cls_idx = torch.max(image_pred[:,5:5+ num_classes], 1)
        cls_idx = cls_idx.float().unsqueeze(1)
        prob = prob.float().unsqueeze(1) # 添加一个轴
        
        seq = (image_pred[:,:5], prob, cls_idx) 
        image_pred = torch.cat(seq, 1) # 按照第一个轴组合起来（横向组合）
    
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        #Get the various classes detected in the image
        img_classes = torch.unique(image_pred_[:,-1])
     
        #WE will do NMS classwise
        output = []
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()  
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            if nms:
                idxs = non_max_sup(image_pred_class, nms_conf)
                image_pred_class = image_pred_class[idxs, :]
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            output.append(torch.cat(seq,1))
    return torch.cat(output)


def plot_rectangle(out, images, classes, colors, line_thickness=None):
    """Add label and rectangle to an image.
    Parameters
    --------
    out : A vector, represent output of network, out=[batch_num,x1,y1,x2,y2,score,prob,cls_num].
    images : images tensor, images[i] = [H, W, C].
    """
    c1 = tuple(out[1:3].int())
    c2 = tuple(out[3:5].int())
    img = images[int(out[0])]
    cls = int(out[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))
    cv2.rectangle(img, c1, c2,color, thickness=tl)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img



