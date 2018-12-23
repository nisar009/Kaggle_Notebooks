import cv2
import numpy as np

from fastai.vision.image import *


# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_4_channel(fname, suffix='.png'):
  
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
        suffix = '.png'
    elif fname.endswith('.jpg'):
        fname = fname[:-4]
        suffix = '.jpg'
    
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    try:
        img = [cv2.imread(fname+'_'+color+suffix, flags).astype(np.float32)/255
               for color in colors]
    except:
        print(fname+suffix)
        return
    
    x = np.stack(img, axis=-1)
    return Image(pil2tensor(x, np.float32).float())