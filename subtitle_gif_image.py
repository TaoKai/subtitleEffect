import cv2
import numpy as np
from six import string_types, iteritems
from PIL import Image, ImageDraw, ImageFont
from single_character import SubAnimation, ArrayMultiline, saveVideo
from module_logger import logger

def get_background(image):
    crop_size = (540, 960)
    # image = cv2.GaussianBlur(image, (9,9), 0)
    # h, w, _ = image.shape
    # p = [int(w/2.0-crop_size[0]/2.0), int(h/2.0-crop_size[1]/2.0)]
    # p[0] = np.clip(p[0], 0, crop_size[0])
    # p[1] = np.clip(p[1], 0, crop_size[1])
    # image = image[p[1]:p[1]+crop_size[1], p[0]:p[0]+crop_size[0], :]
    image = cv2.resize(image, crop_size, interpolation=cv2.INTER_AREA)
    return image

class GIFPIC(object):
    def __init__(self, path):
        self.cursor = 0
        self.gif_path = path
        self.gif_frames = self.get_gif_frames()
        self.gif_len = len(self.gif_frames)
    
    def get_gif_frames(self):
        gifs = Image.open(self.gif_path)
        frames = []
        for i in range(gifs.n_frames):
            gifs.seek(i)
            new = Image.new("RGBA", gifs.size)
            new.paste(gifs)
            img = np.array(new)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            frames.append(img)
        return frames
    
    def get_loop_image(self, shape=None):
        img = self.gif_frames[self.cursor]
        if shape is not None:
            img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
        self.cursor = (self.cursor+1)%self.gif_len
        if img.shape[2]==4:
            frame = img[:,:,:3]
            gif_map = img[:,:,3]/255.0
            gif_map = np.array([gif_map, gif_map, gif_map]).transpose(1, 2, 0)
        else:
            frame = img
            gif_map = np.ones(frame.shape, dtype=np.float32)
        return frame, gif_map
    
    def draw_on_image(self, image, position, shape=None):
        img = image.copy()
        line_img, line_map = self.get_loop_image(shape)
        width = line_img.shape[1]
        height = line_img.shape[0]
        p = position
        p0 = [int(p[0]-width/2.0), int(p[1]-height/2)]
        p1 = [p0[0]+width, p0[1]+height]
        img_p0 = [0,0]
        img_p1 = [img.shape[1], img.shape[0]]
        crop_p0 = [max(p0[0], img_p0[0]), max(p0[1], img_p0[1])]
        crop_p1 = [min(p1[0], img_p1[0]), min(p1[1], img_p1[1])]
        line_p0 = [crop_p0[0]-p0[0], crop_p0[1]-p0[1]]
        line_p1 = [crop_p1[0]-p0[0], crop_p1[1]-p0[1]]
        line_img = line_img[line_p0[1]:line_p1[1], line_p0[0]:line_p1[0], :]
        line_map = line_map[line_p0[1]:line_p1[1], line_p0[0]:line_p1[0], :]
        if line_img.shape[0]<=0 or line_img.shape[1]<=0:
            return img
        line_map = cv2.GaussianBlur(line_map, (5,5), 0)
        cut = img[crop_p0[1]:crop_p1[1], crop_p0[0]:crop_p1[0], :]
        if cut.shape[0]<=0 or cut.shape[1]<=0:
            return img
        blend = line_img*line_map+cut*(1-line_map)
        blend = blend.astype(np.uint8)
        img[crop_p0[1]:crop_p1[1], crop_p0[0]:crop_p1[0], :] = blend
        return img

class GifSubAnimation(SubAnimation):
    def __init__(self, para_string, font_size):
        super(GifSubAnimation, self).__init__(para_string, font_size)
        self.make_combination = self.make_image_gif_combination

    def merge_image_gif(self, image, gifpic):
        h, w, _ = image.shape
        img = gifpic.draw_on_image(image, (w/2,h/2+250), (512, 512))
        return img

    def make_margin_and_fade_animation_with_image_gif(self, image, gifpic, margin_range=(0,0), alpha_range=(1.0,0.0)):
        h, w, _ = image.shape
        position = (w/2, h/2-100)
        alpha = alpha_range[0]
        intv_alpha = (alpha_range[0]-alpha_range[1])/self.fade_frame_cnt
        margin = margin_range[0]
        intv_margin = (margin_range[0]-margin_range[1])/self.fade_frame_cnt
        frames = []
        for i in range(self.fade_frame_cnt):
            gif_img = self.merge_image_gif(image, gifpic)
            am = ArrayMultiline(self.para_string, self.font_size, alpha=alpha, margin=margin)
            img = am.draw_on_image(gif_img, position)
            alpha -= intv_alpha
            margin -= intv_margin
            frames.append(img)
            logger.info(f'make_margin_and_fade_animation_with_image {i}')
        gif_img = self.merge_image_gif(image, gifpic)
        am = ArrayMultiline(self.para_string, self.font_size, alpha=alpha_range[1], margin=margin_range[1])
        img = am.draw_on_image(gif_img, position)
        frames.append(img)
        return frames

    def make_scale_animation_with_image_gif(self, image, gifpic, scale_ratio=3.0):
        orig_img = image.copy()
        h, w, _ = orig_img.shape
        position = (w/2, h/2-100)
        start_size = int(self.font_size*scale_ratio)
        intv = (start_size-self.font_size)/self.scale_frame_cnt
        frames = []
        for i in range(self.scale_frame_cnt):
            img = self.merge_image_gif(orig_img, gifpic)
            am = ArrayMultiline(self.para_string, int(start_size))
            img = am.draw_on_image(img, position)
            start_size -= intv
            frames.append(img)
            logger.info(f'make_scale_animation_with_image_gif {i}')
        img = self.merge_image_gif(orig_img, gifpic)
        am = ArrayMultiline(self.para_string, self.font_size)
        img = am.draw_on_image(img, position)
        frames.append(img)
        last = orig_img
        last = am.draw_on_image(last, position)
        return frames, last

    def make_pause_animation_with_image_gif(self, image, gifpic, has_character=True):
        if has_character:
            frame_cnt = self.pause_per_character_frame_cnt*self.p_len
        else:
            frame_cnt = self.pause_no_character_frame_cnt
        frames = []
        for i in range(frame_cnt):
            img = self.merge_image_gif(image, gifpic)
            frames.append(img)
            logger.info(f'make_pause_animation_with_image_gif {i} {has_character}')
        return frames
    
    def make_image_gif_combination(self, image, gifpic):
        frames = self.make_pause_animation_with_image_gif(image, gifpic, has_character=False)
        start_index = len(frames)
        sframes, last_frame = self.make_scale_animation_with_image_gif(image, gifpic, scale_ratio=3.0)
        frames += sframes
        frames += self.make_pause_animation_with_image_gif(last_frame, gifpic)
        frames += self.make_margin_and_fade_animation_with_image_gif(image, gifpic, margin_range=(0, 40), alpha_range=(1.0, 0.0))
        frames += self.make_pause_animation_with_image_gif(image, gifpic, has_character=False)
        return frames, start_index

class SubtitleGifAdvertise(object):
    def __init__(self, lines):
        self.subAnims = []
        for l in lines:
            gsa = GifSubAnimation(l, 80)
            self.subAnims.append(gsa)
    
    def make_advertisement(self, images, gifpic):
        images = [get_background(img) for img in images]
        img_cnt = len(images)
        sa_cnt = len(self.subAnims)
        if sa_cnt<img_cnt:
            img_cnt = sa_cnt
        if sa_cnt%img_cnt == 0:
            intv = int(sa_cnt/img_cnt)
        else:
            intv = int(sa_cnt/img_cnt+1)
        img_ind = 0
        image = images[img_ind]
        frames = []
        frame_indices = []
        for i, sa in enumerate(self.subAnims):
            if i>0 and i%intv==0 and img_ind<img_cnt-1:
                img_ind += 1
                image = images[img_ind]
            paraframes, paraindex = sa.make_combination(image, gifpic)
            paraindex += len(frames)
            frames += paraframes
            frame_indices.append(paraindex)
        return frames, frame_indices

def convert2millisecond(frame_indices, fps=30):
    millis = [int(fi*1000/fps) for fi in frame_indices]
    return millis

def generate_gif_image_advertisement(save_path, lines, img_paths, gif_path):
    gifpic = GIFPIC(gif_path)
    images = [cv2.imread(p, cv2.IMREAD_COLOR) for p in img_paths]
    for im in images:
        if im is None:
            print('Exists none image!')
            return None, None
    sadv = SubtitleGifAdvertise(lines)
    frames, frame_indices = sadv.make_advertisement(images, gifpic)
    millis = convert2millisecond(frame_indices)
    lines_delay = []
    for l, m in zip(lines, millis):
        lines_delay.append((l, m))
    saveVideo(save_path, frames, 30)
    print('Video is saved to', save_path)
    return frames, lines_delay

if __name__=='__main__':
    print('start animation.')
    basic = 'E:/workspace/subtitleEffect/'
    img_paths = [
        basic+'test.png',
        basic+'test1.png',
        basic+'test2.png',
        basic+'test3.png'
    ]
    lines = ['因为牛郎和织女一直处于分居状态', '牛郎和他的牛好上了', '所以七夕节不过了', '请大家相互转告', '如果还想过的话', '就请集齐七颗龙珠', '这样就可以召唤神龙了']
    frames, _ = generate_gif_image_advertisement('tmp.mp4', lines, img_paths, basic+'boat.gif')
    for im in frames:
        cv2.imshow('', im)
        cv2.waitKey(int(1000/30))
    cv2.waitKey(0)