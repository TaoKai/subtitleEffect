from single_character import generate_advertisement
import cv2

if __name__=='__main__':
    paths = ['test.png','test1.png','test2.png','test3.png']
    lines = ['因为牛郎和织女一直处于分居状态', '牛郎和他的牛好上了', '所以七夕节不过了', '请大家相互转告', '如果还想过的话', '就请集齐七颗龙珠', '这样就可以召唤神龙了']
    frames, lines_delay = generate_advertisement('tmp.mp4', lines, paths)
    for l, m in lines_delay:
        print(l, m)
    for img in frames:
        cv2.imshow('', img)
        cv2.waitKey(int(1000/30))