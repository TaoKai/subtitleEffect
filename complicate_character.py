import cv2
import numpy as np
from six import string_types, iteritems
from PIL import Image, ImageDraw, ImageFont
from codecs import open
import random

def rotate_center(img, degree):
    h, w, _ = img.shape
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), degree, 1)
    img = cv2.warpAffine(img, mat, (w, h), borderMode=0)
    return img

def get_hanzi_string():
    h_string = open('hanzi.txt', 'r', 'utf-8').read().strip()
    return h_string

class ComplicateCharacter(object):
    def __init__(self, ch_string, font_size, rot_angle=0, alpha=1.0, inner_color=(255,255,255), outter_color=(30,30,30)):
        if font_size <=5:
            font_size = 5
        self.ch_string = ch_string[0]
        self.font_size = font_size
        self.inner_font_size = int(font_size*0.97)
        self.inner_color = inner_color
        self.outter_color = outter_color
        self.alpha = alpha
        self.rot_angle = rot_angle
        self.ch_img_size = int(self.font_size*np.sqrt(2))
    
    def get_center(self, position):
            inner_p = [int(x-self.inner_font_size/2.0) for x in position]
            outter_p = [int(x-self.font_size/2.0) for x in position]
            return inner_p, outter_p

    def draw_at_raw(self, img, position):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        _, op = self.get_center(position)
        pilImg = Image.fromarray(img)
        draw = ImageDraw.Draw(pilImg)
        font = ImageFont.truetype("simhei.ttf", self.font_size, encoding="utf-8")
        draw.text(op, self.ch_string, self.outter_color, font=font)
        font = ImageFont.truetype("simhei.ttf", self.inner_font_size, encoding="utf-8")
        draw.text(op, self.ch_string, self.inner_color, font=font)
        img = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
        return img

    def get_inner_draw(self):
        ch_img = np.zeros((self.ch_img_size, self.ch_img_size, 3), dtype=np.uint8)
        h, w, _ = ch_img.shape
        ch_img = self.draw_at_raw(ch_img, (w/2, h/2))
        ch_img = rotate_center(ch_img, self.rot_angle)
        ch_map = np.where(ch_img>0, np.ones(ch_img.shape, np.float32), np.zeros(ch_img.shape, np.float32))*self.alpha
        ch_map = np.clip(ch_map, 0, 1)
        return ch_img, ch_map
    
    def draw_at(self, image, position):
        img = image.copy()
        line_img, line_map = self.get_inner_draw()
        p = position
        p0 = [int(p[0]-self.ch_img_size/2.0), int(p[1]-self.ch_img_size/2)]
        p1 = [p0[0]+self.ch_img_size, p0[1]+self.ch_img_size]
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

class SingleCharacterAnim(object):
    def __init__(self, ch_string, font_size):
        self.ch_string = ch_string
        self.font_size = font_size
        self.hanzi = get_hanzi_string()
        self.rain_single_anim_cnt = 15
        self.rain_time_cnt = 4
        self.rain_intv = 12
        self.single_total_cnt = self.rain_intv*(self.rain_time_cnt-1)+self.rain_single_anim_cnt
        self.rain_list, self.rain_total_cnt = self.get_string_rain_list()
    
    def get_string_positions(self, from_pos, to_pos):
        w = len(self.ch_string)*self.font_size
        from_x = from_pos[0]-w/2+self.font_size/2
        to_x = to_pos[0]-w/2+self.font_size/2
        string_positions = [((from_x+i*self.font_size, from_pos[1]),(to_x+i*self.font_size, to_pos[1])) for i in range(len(self.ch_string))]
        return string_positions

    def get_string_rain_list(self):
        str_cnt = len(self.ch_string)
        string_rain_list = [self.get_single_rain_list(start_ind=random.randint(0, 20)) for _ in range(str_cnt)]
        max_ind = np.max(np.array(string_rain_list, dtype=np.int32))
        return string_rain_list, max_ind

    def get_single_rain_list(self, start_ind=0):
        rList = []
        for _ in range(self.rain_time_cnt):
            rrange = (start_ind, start_ind+self.rain_single_anim_cnt)
            rList.append(rrange)
            start_ind = start_ind+self.rain_intv
        return rList

    def rain_move_character_in_image(self, image, from_pos, to_pos, alpha_range, rot_range):
        string_positions = self.get_string_positions(from_pos, to_pos)
        dr = (rot_range[1]-rot_range[0])/self.rain_single_anim_cnt
        dx = (to_pos[0]-from_pos[0])/self.rain_single_anim_cnt
        dy = (to_pos[1]-from_pos[1])/self.rain_single_anim_cnt
        da = (alpha_range[1]-alpha_range[0])/self.single_total_cnt
        start_a = [alpha_range[0] for _ in range(len(self.ch_string))]
        start_xyr = [[[sp[0][0], sp[0][1], rot_range[0]] for _ in range(self.rain_time_cnt)] for sp in string_positions]
        frames = []
        index = [random.randint(0, len(self.ch_string)-1) for _ in range(len(self.ch_string))]
        for i in range(self.rain_total_cnt):
            frame_img = image
            for j, rl in enumerate(self.rain_list):
                for k, sr in enumerate(rl):
                    if i in range(sr[0], sr[1]):
                        if i%2==0:
                            index[j] = random.randint(0, len(self.ch_string)-1)
                        cc = ComplicateCharacter(self.ch_string[index[j]], self.font_size, alpha=start_a[j], rot_angle=start_xyr[j][k][2])
                        frame_img = cc.draw_at(frame_img, (start_xyr[j][k][0], start_xyr[j][k][1]))
                        start_xyr[j][k][0] += dx
                        start_xyr[j][k][1] += dy
                        start_xyr[j][k][2] += dr
                if i>=rl[0][1]:
                    cc = ComplicateCharacter(self.ch_string[j], self.font_size, alpha=start_a[j], rot_angle=0)
                    frame_img = cc.draw_at(frame_img, (string_positions[j][1][0], string_positions[j][1][1]))
                if i in range(rl[0][0], rl[-1][1]):
                    start_a[j] += da
            frames.append(frame_img)
            print('rain_move_character_in_image', i)
        img = image
        for i in range(len(self.ch_string)):
            cc = ComplicateCharacter(self.ch_string[i], self.font_size, alpha=alpha_range[1], rot_angle=0)
            img = cc.draw_at(img, (string_positions[i][1][0], string_positions[i][1][1]))
        frames.append(img)
        return frames

if __name__=='__main__':
    hanzi = get_hanzi_string()
    img = cv2.imread('test.png', cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    sca = SingleCharacterAnim('好好学习天天向上', 100)
    frames = sca.rain_move_character_in_image(img, (w/2,0), (w/2, 600), (0, 1), (10, 0))
    cv2.imshow('', img)
    cv2.waitKey(0)
    for f in frames:
        cv2.imshow('', f)
        cv2.waitKey(int(1000/30))
    cv2.waitKey(0)
    