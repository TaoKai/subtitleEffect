import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from skimage.measure import label

clip = VideoFileClip('bface.gif', has_mask=True) # can be gif or movie
for frame in clip.iter_frames():
    # now frame is a numpy array, do wathever you want
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32)
    mask = (frame[:,:,0]+frame[:,:,1]+frame[:,:,2])/(255*3)
    mask[mask<1] = 0
    # labeled_img, num = label(mask, neighbors=4, background=0, return_num=True) 
    # max_list = []
    # for i in range(1, num+1):
    #     total = np.sum(labeled_img==i)
    #     max_list.append(total)
    # max_ind = np.argmax(np.array(max_list))+1
    # labeled_img[labeled_img!=max_ind] = 0
    # mask = labeled_img.astype(np.float32)/max_ind
    mask = 1-mask
    mask = np.array([mask, mask, mask]).transpose(1,2,0)
    bg = np.ones(frame.shape, dtype=np.float32)*150
    frame = frame*mask+bg*(1-mask)
    print(frame.shape)
    cv2.imshow('', frame.astype(np.uint8))
    cv2.waitKey(0)
    cv2.waitKey(33)
    