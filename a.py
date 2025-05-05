from HiFi_Net import HiFi_Net 
from PIL import Image
import numpy as np

HiFi = HiFi_Net()   # initialize
img_path = '/home/sidx/myDrive/internship/imageForgeryDetection/HiFi_IFDL/data_dir/myDatasetImage/tulip.jpg'

## detection
res3, prob3 = HiFi.detect(img_path)
# print(res3, prob3) 1 1.0
HiFi.detect(img_path, verbose=True)

## localization
binary_mask = HiFi.localize(img_path)
binary_mask = Image.fromarray((binary_mask*255.).astype(np.uint8))
binary_mask.save('pred_mask.png')