import cv2
import numpy as np
from six import string_types, iteritems
from PIL import Image, ImageDraw, ImageFont
from single_character import SubAnimation, ArrayMultiline, saveVideo
from module_logger import logger
from subtitle_gif_image import GIFPIC

def get_background(image):
    crop_size = (540, 960)
    image = cv2.resize(image, crop_size, interpolation=cv2.INTER_AREA)
    return image

def show(img, time=0):
    cv2.imshow('', img)
    cv2.waitKey(time)

class VIDEO(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = self.prepare_video_frames()
        self.cursor = 0
    
    def center_crop(self, frame):
        h, w, _ = frame.shape
        if h/w<=16/9:
            new_w = h/16*9
            x = w/2-new_w/2
            x = int(x)
            new_w = int(new_w)
            frame = frame[:, x:x+new_w, :]
        else:
            new_h = w/9*16
            y = h/2-new_h/2
            y = int(y)
            new_h = int(new_h)
            frame = frame[y:y+new_h, :, :]
        frame = get_background(frame)
        return frame

    def prepare_video_frames(self, num=500):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        for i in range(num):
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = self.center_crop(frame)
                    frames.append(frame)
                    logger.info(f'prepare_video_frames {i}')
                else:
                    break
        if len(frames)<=0:
            frames.append(np.zeros([960, 540, 3], np.uint8))
        return frames
    
    def get_loop_image(self):
        image = self.frames[self.cursor]
        self.cursor = (self.cursor+1)%len(self.frames)
        return image

class GifVideoSubAnimation(SubAnimation):
    def __init__(self, para_string, font_size):
        super(GifVideoSubAnimation, self).__init__(para_string, font_size)
        self.make_combination = self.make_video_gif_combination
    
    def make_video_gif_combination(self, video, gifpic):
        frames = self.make_pause_animation_with_video_gif(video, gifpic, has_character=False)
        start_index = len(frames)
        sframes = self.make_scale_animation_with_video_gif(video, gifpic, scale_ratio=3.0)
        frames += sframes
        frames += self.make_pause_animation_with_video_gif(video, gifpic)
        frames += self.make_margin_and_fade_animation_with_video_gif(video, gifpic, margin_range=(0, 40), alpha_range=(1.0, 0.0))
        frames += self.make_pause_animation_with_video_gif(video, gifpic, has_character=False)
        return frames, start_index

    def merge_image_gif(self, video, gifpic):
        image = video.get_loop_image()
        h, w, _ = image.shape
        img = gifpic.draw_on_image(image, (w/2,h/2+250), (512, 512))
        return img
    
    def make_pause_animation_with_video_gif(self, video, gifpic, has_character=True):
        if has_character:
            frame_cnt = self.pause_per_character_frame_cnt*self.p_len
        else:
            frame_cnt = self.pause_no_character_frame_cnt
        frames = []
        for i in range(frame_cnt):
            img = self.merge_image_gif(video, gifpic)
            h, w, _ = img.shape
            if has_character:
                am = ArrayMultiline(self.para_string, self.font_size, alpha=1, margin=0)
                img = am.draw_on_image(img, (w/2,h/2-100))
            frames.append(img)
            logger.info(f'make_pause_animation_with_video_gif {i} {has_character}')
        return frames
    
    def make_scale_animation_with_video_gif(self, video, gifpic, scale_ratio=3.0):
        start_size = int(self.font_size*scale_ratio)
        intv = (start_size-self.font_size)/self.scale_frame_cnt
        frames = []
        h, w = 0, 0
        for i in range(self.scale_frame_cnt):
            img = self.merge_image_gif(video, gifpic)
            h, w, _ = img.shape
            position = (w/2, h/2-100)
            am = ArrayMultiline(self.para_string, int(start_size))
            img = am.draw_on_image(img, position)
            start_size -= intv
            frames.append(img)
            logger.info(f'make_scale_animation_with_video_gif {i}')
        position = (w/2, h/2-100)
        img = self.merge_image_gif(video, gifpic)
        am = ArrayMultiline(self.para_string, self.font_size)
        img = am.draw_on_image(img, position)
        frames.append(img)
        return frames

    def make_margin_and_fade_animation_with_video_gif(self, video, gifpic, margin_range=(0,40), alpha_range=(1.0,0.0)):
        alpha = alpha_range[0]
        intv_alpha = (alpha_range[0]-alpha_range[1])/self.fade_frame_cnt
        margin = margin_range[0]
        intv_margin = (margin_range[0]-margin_range[1])/self.fade_frame_cnt
        frames = []
        for i in range(self.fade_frame_cnt):
            gif_img = self.merge_image_gif(video, gifpic)
            h, w, _ = gif_img.shape
            position = (w/2, h/2-100)
            am = ArrayMultiline(self.para_string, self.font_size, alpha=alpha, margin=margin)
            img = am.draw_on_image(gif_img, position)
            alpha -= intv_alpha
            margin -= intv_margin
            frames.append(img)
            logger.info(f'make_margin_and_fade_animation_with_video_gif {i}')
        gif_img = self.merge_image_gif(video, gifpic)
        h, w, _ = gif_img.shape
        position = (w/2, h/2-100)
        am = ArrayMultiline(self.para_string, self.font_size, alpha=alpha_range[1], margin=margin_range[1])
        img = am.draw_on_image(gif_img, position)
        frames.append(img)
        return frames

class VideoGifSubAdvertise(object):
    def __init__(self, lines):
        self.subAnims = []
        for l in lines:
            vsa = GifVideoSubAnimation(l, 80)
            self.subAnims.append(vsa)
    
    def make_advertisement(self, video, gifpic):
        frames = []
        frame_indices = []
        for _, sa in enumerate(self.subAnims):
            paraframes, paraindex = sa.make_combination(video, gifpic)
            paraindex += len(frames)
            frames += paraframes
            frame_indices.append(paraindex)
        return frames, frame_indices

def convert2millisecond(frame_indices, fps=30):
    millis = [int(fi*1000/fps) for fi in frame_indices]
    return millis

def generate_gif_video_advertisement(save_path, lines, video_path, gif_path):
    gifpic = GIFPIC(gif_path)
    video = VIDEO(video_path)
    sadv = VideoGifSubAdvertise(lines)
    frames, frame_indices = sadv.make_advertisement(video, gifpic)
    millis = convert2millisecond(frame_indices)
    lines_delay = []
    for l, m in zip(lines, millis):
        lines_delay.append((l, m))
    saveVideo(save_path, frames, 30)
    print('Video is saved to', save_path)
    return frames, lines_delay

if __name__=='__main__':
    print('start animation.')
    lines = ['因为牛郎和织女一直处于分居状态', '牛郎和他的牛好上了', '所以七夕节不过了', '请大家相互转告', '如果还想过的话', '就请集齐七颗龙珠', '这样就可以召唤神龙了']
    frames, frames_indices = generate_gif_video_advertisement('tmp.mp4', lines, 'scenery.mp4', 'xiuwu.gif')
    for fi in frames_indices:
        print(fi)
    for fr in frames:
        show(fr, 33)
