import sys 
sys.path.append('.')
import torch
from yolo.darknet import Darknet
from yolo.io import load_classes, load_colors
from yolo.util import sift_results
import random
import cv2

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

class DarknetModel(object):
    def __init__(self):
        self.scales = "1,2,3"
        self.batch_size = 1
        self.confidence = 0.5
        self.nms_thesh = 0.4
        self.reso = 416
        self.CUDA = False
        self.num_classes = 80
        self.classes = load_classes('data/coco.names') 
        self.colors = load_colors('data/pallete')
        self.model = Darknet('cfg/yolov3.cfg')
        self.model.load_state_dict(torch.load('yolov3.pkl'))
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0 
        assert self.inp_dim > 32
        if self.CUDA:
            self.model.cuda()
        self.model.eval()
    def predict(self, filename):
        image = cv2.imread(filename)
        img, orig_im, dim = prep_image(image, self.inp_dim)  
        im_dim = torch.FloatTensor(dim).repeat(1,2)        
        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        output = self.model(img)
        output = sift_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thesh)
        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(self.inp_dim))/self.inp_dim
            
        output[:,[1,3]] *= image.shape[1]
        output[:,[2,4]] *= image.shape[0]

        print(output.shape)
        list(map(lambda x: write(x, orig_im, self.classes, self.colors), output))
        return orig_im
