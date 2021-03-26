import cv2
import numpy as np
from six import string_types, iteritems
from PIL import Image, ImageDraw, ImageFont

def saveVideo(path, frames, frate):
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frate, (frames[0].shape[1],frames[0].shape[0]))
    for f in frames:
        videoWriter.write(f)
    videoWriter.release()

class SingleCharacter(object):
    def __init__(self, ch_string, font_size, inner_color, outter_color):
        self.ch_string = ch_string
        if font_size <=14:
            font_size = 14
        self.font_size = font_size
        self.inner_font_size = int(font_size*0.97)
        self.inner_color = inner_color
        self.outter_color = outter_color
    
    def get_center(self, position):
        inner_p = [int(x-self.inner_font_size/2.0) for x in position]
        outter_p = [int(x-self.font_size/2.0) for x in position]
        return inner_p, outter_p

    def draw_at(self, img, position):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        ip, op = self.get_center(position)
        pilImg = Image.fromarray(img)
        draw = ImageDraw.Draw(pilImg)
        font = ImageFont.truetype("simhei.ttf", self.font_size, encoding="utf-8")
        draw.text(op, self.ch_string, self.outter_color, font=font)
        font = ImageFont.truetype("simhei.ttf", self.inner_font_size, encoding="utf-8")
        draw.text(op, self.ch_string, self.inner_color, font=font)
        img = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
        return img

class ArrayCharacter(object):
    def __init__(self, line_string, font_size, inner_color, outter_color, alpha, margin=0):
        if font_size <=14:
            font_size = 14
        self.font_size = font_size
        self.line_string = line_string
        self.c_len = len(self.line_string)
        self.characters = [SingleCharacter(ch, self.font_size, inner_color, outter_color) for ch in self.line_string]
        self.alpha = np.clip(alpha, 0, 1)
        self.margin = margin
        self.width = int(self.c_len*(self.font_size+self.margin*2))
        self.height = int(self.margin*2+self.font_size)
    
    def draw_on_image(self, image, position):
        img = image.copy()
        line_img, line_map = self.draw_line_and_map()
        p = position
        p0 = [int(p[0]-self.width/2.0), int(p[1]-self.height/2)]
        p1 = [p0[0]+self.width, p0[1]+self.height]
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
        
    def draw_line_and_map(self):
        lineImg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pos = (self.width/2.0, self.height/2.0)
        lineImg = self.draw_at(lineImg, pos)
        lineMap = np.where(lineImg>0, np.ones((self.height, self.width, 3), dtype=np.float32), np.zeros((self.height, self.width, 3), dtype=np.float32))
        lineMap = lineMap*self.alpha
        return lineImg, lineMap

    def draw_at(self, img, position):
        p = position
        start_p = [int(p[0]-self.width/2.0+self.height/2.0), int(p[1])]
        for i in range(self.c_len):
            sc = self.characters[i]
            img = sc.draw_at(img, start_p)
            start_p[0] += self.height
        return img

class ArrayMultiline(object):
    def __init__(self, line_string, font_size, inner_color=(255,255,255), outter_color=(10,10,10), alpha=1.0, margin=0):
        if font_size <=14:
            font_size = 14
        self.margin = margin
        self.font_size = font_size
        self.line_string = line_string
        self.ls_len = len(self.line_string)
        self.alpha = np.clip(alpha, 0, 1)
        self.line1len = 6
        self.line2len = 7
        self.lines = self.clip_lines()
        self.height = len(self.lines)*(self.font_size+self.margin*2)
        self.array_lines = []
        for l in self.lines:
            ach = ArrayCharacter(l, self.font_size, inner_color, outter_color, alpha, margin)
            self.array_lines.append(ach)
    
    def draw_on_image(self, image, position):
        p = position
        start_p = [p[0], p[1]-self.height/2.0+self.font_size/2.0+self.margin]
        for ach in self.array_lines:
            image = ach.draw_on_image(image, start_p)
            start_p[1] += self.font_size+2*self.margin
        return image

    def clip_lines(self):
        lines = []
        line_str = self.line_string
        if self.ls_len<=self.line1len:
            lines.append(line_str)
            return lines
        else:
            start_index = self.line1len
            lines.append(line_str[:self.line1len])
            while start_index<self.ls_len:
                c_str = line_str[start_index:start_index+self.line2len]
                if len(c_str)<=2:
                    tmp_str = lines[-1]
                    lines[-1] = tmp_str[:-2]
                    c_str = tmp_str[-2:]+c_str
                lines.append(c_str)
                start_index += self.line2len
        return lines

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

class SubAnimation(object):
    def __init__(self, para_string, font_size):
        self.make_combination = self.make_combination02
        if font_size <=14:
            font_size = 14
        self.font_size = font_size
        self.para_string = para_string
        self.scale_frame_cnt = 5
        self.fade_frame_cnt = 5
        self.p_len = len(self.para_string)
        self.pause_per_character_frame_cnt = 5
        self.pause_no_character_frame_cnt = 5
    
    def make_margin_and_fade_animation_with_image(self, image, margin_range=(0,0), alpha_range=(1.0,0.0)):
        h, w, _ = image.shape
        position = (w/2, h/2+80)
        alpha = alpha_range[0]
        intv_alpha = (alpha_range[0]-alpha_range[1])/self.fade_frame_cnt
        margin = margin_range[0]
        intv_margin = (margin_range[0]-margin_range[1])/self.fade_frame_cnt
        frames = []
        for i in range(self.fade_frame_cnt):
            am = ArrayMultiline(self.para_string, self.font_size, alpha=alpha, margin=margin)
            img = am.draw_on_image(image, position)
            alpha -= intv_alpha
            margin -= intv_margin
            frames.append(img)
            print('make_margin_and_fade_animation_with_image', i)
        am = ArrayMultiline(self.para_string, self.font_size, alpha=alpha_range[1], margin=margin_range[1])
        img = am.draw_on_image(image, position)
        frames.append(img)
        return frames

    def make_fade_animation_with_image(self, image, alpha_range=(1.0,0.0)):
        h, w, _ = image.shape
        position = (w/2, h/2+80)
        alpha = alpha_range[0]
        intv = (alpha_range[0]-alpha_range[1])/self.fade_frame_cnt
        frames = []
        for i in range(self.fade_frame_cnt):
            am = ArrayMultiline(self.para_string, self.font_size, alpha=alpha)
            img = am.draw_on_image(image, position)
            alpha -= intv
            frames.append(img)
            print('make_fade_animation_with_image', i)
        am = ArrayMultiline(self.para_string, self.font_size, alpha=alpha_range[1])
        img = am.draw_on_image(image, position)
        frames.append(img)
        return frames

    def make_pause_animation_with_image(self, image, has_character=True):
        if has_character:
            frame_cnt = self.pause_per_character_frame_cnt*self.p_len
        else:
            frame_cnt = self.pause_no_character_frame_cnt
        frames = []
        for i in range(frame_cnt):
            frames.append(image)
            print('make_pause_animation_with_image', i, has_character)
        return frames

    def make_scale_animation_with_image(self, image, scale_ratio=3.0):
        orig_img = image.copy()
        h, w, _ = orig_img.shape
        position = (w/2, h/2+80)
        start_size = int(self.font_size*scale_ratio)
        intv = (start_size-self.font_size)/self.scale_frame_cnt
        frames = []
        for i in range(self.scale_frame_cnt):
            am = ArrayMultiline(self.para_string, int(start_size))
            img = am.draw_on_image(orig_img, position)
            start_size -= intv
            frames.append(img)
            print('make_scale_animation_with_image', i)
        am = ArrayMultiline(self.para_string, self.font_size)
        img = am.draw_on_image(orig_img, position)
        frames.append(img)
        return frames
    
    def make_combination01(self, image):
        frames = self.make_pause_animation_with_image(image, has_character=False)
        start_index = len(frames)
        frames += self.make_scale_animation_with_image(image, scale_ratio=3.0)
        frames += self.make_pause_animation_with_image(frames[-1])
        frames += self.make_fade_animation_with_image(image, alpha_range=(1.0, 0.0))
        frames += self.make_pause_animation_with_image(image, has_character=False)
        return frames, start_index
    
    def make_combination02(self, image):
        frames = self.make_pause_animation_with_image(image, has_character=False)
        start_index = len(frames)
        frames += self.make_scale_animation_with_image(image, scale_ratio=3.0)
        frames += self.make_pause_animation_with_image(frames[-1])
        frames += self.make_margin_and_fade_animation_with_image(image, margin_range=(0, 40), alpha_range=(1.0, 0.0))
        frames += self.make_pause_animation_with_image(image, has_character=False)
        return frames, start_index

class SubtitleAdvertise(object):
    def __init__(self, lines):
        self.subAnims = []
        for l in lines:
            sa = SubAnimation(l, 80)
            self.subAnims.append(sa)
    
    def make_advertisement(self, images):
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
            paraframes, paraindex = sa.make_combination(image)
            paraindex += len(frames)
            frames += paraframes
            frame_indices.append(paraindex)
        return frames, frame_indices

def convert2millisecond(frame_indices, fps=30):
    millis = [int(fi*1000/fps) for fi in frame_indices]
    return millis

def generate_advertisement(save_path, lines, img_paths):
    images = [cv2.imread(p, cv2.IMREAD_COLOR) for p in img_paths]
    for im in images:
        if im is None:
            print('Exists none image!')
            return None, None
    sadv = SubtitleAdvertise(lines)
    frames, frame_indices = sadv.make_advertisement(images)
    millis = convert2millisecond(frame_indices)
    lines_delay = []
    for l, m in zip(lines, millis):
        lines_delay.append((l, m))
    saveVideo(save_path, frames, 30)
    print('Video is saved to', save_path)
    return frames, lines_delay

if __name__=='__main__':
    image = cv2.imread('test.png', cv2.IMREAD_COLOR)
    image = get_background(image)
    h, w, _ = image.shape
    line_string = '因为牛郎和织女一直处于分居状态'
    am = ArrayMultiline(line_string, 60, margin=5)
    img = am.draw_on_image(image, (w/2, h/2+80))
    cv2.imshow('', img)
    cv2.waitKey(0)