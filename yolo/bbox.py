import torch


def bbox_iou(box1, box2, cuda=True):
    """Returns the IoU of two bounding boxes 
    Parameters
    ----------
    box1 : the first bounding box, shape=[n_box, coor], coor=(x1, y1, x2, y2).
    box2 : the second bounding box, shape=[n_box, coor], coor=(x1, y1, x2, y2).

    Return
    ----------
    IoU of two bounding boxes, shape=[n_box].
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    #Intersection area
    if cuda and torch.cuda.is_available():
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda(
        ))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * \
            torch.max(inter_rect_y2 - inter_rect_y1 + 1,
                      torch.zeros(inter_rect_x2.shape))

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou
