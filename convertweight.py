import torch
from yolo.darknet import Darknet

def gen_torch_weight(infname, outfname):
    """Convert darknet style model file to PyTorch format.
    Paramaters
    ----------
    infname : Darknet style model file(*.weights).
    outfname : PyTorch style model file(*.pkl).
    """
    net = Darknet('cfg/yolov3.cfg')
    net.load_darknet_weights(infname)
    torch.save(net.state_dict(), outfname)

if __name__ == "__main__":
    gen_torch_weight('yolov3.weights', 'yolov3.pkl')