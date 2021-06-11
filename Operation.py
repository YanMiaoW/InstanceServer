import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lollipop.warping.rotate.Core.fdfa import fdfa
import time
import math
global t_str
import os
t_str = ''
global log_list
log_list = []


def to_onnx(input_data, model, name, dynamic=False):
    NEED_TO_ONNX = False
    if NEED_TO_ONNX:
        if dynamic:
            dynamic_axes = {'data': [2, 3]}
            export_params = True
        else:
            dynamic_axes = None
            export_params = False
        torch.onnx.export(model, (input_data,), '/root/%s.onnx' % name,
                          export_params=export_params,
                          input_names=['data'],
                          output_names=['output'],
                          opset_version=11,
                          dynamic_axes=dynamic_axes
                          )
        print('finish. %s' % name)


class TimeLog:
    def __init__(self, name=None, str_c=None):
        self.start = time.time()
        self.name = name if name is not None else 'The process'
        self.str_c = str_c if str_c is not None else '[%s] cost time: %s ms'
        self.time_log = ''
        global t_str
        t_str += '\t'
        self.color_dir = {
            'title': (1, 33),
            'red': (0, 31),
            'blue': (0, 34),
            'green': (0, 32),
            'white': (0, 37),
            'purple': (0, 35),
            'yellow': (0, 33)
        }
        self.color_id = [
            'green',
            'green',
            'blue',
            'purple',
            'yellow',
            'red'
        ]

    def color(self, string_, mode_color=None):
        if mode_color is None:
            mode_color = self.color_dir['title']
        mode, color = mode_color
        if not isinstance(string_, str):
            string_ = '%.2f' % string_
        str_color = '\033[%d;%dm%s\033[0m' % (mode, color, string_)
        return str_color

    def __enter__(self):
        global log_list
        log_list.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = (time.time() - self.start) * 1000
        global t_str
        t_str = t_str[:-1]
        num = int(math.ceil(np.log10(end_time)))
        # print(self.color_id[num])
        self.time_log = t_str + self.str_c % (
            self.color(self.name),
            self.color(end_time, self.color_dir[self.color_id[num]])
        )
        if len(t_str) == 0:
            global log_list
            for log in log_list:
                print(log.time_log)
            log_list = []


def min_filter(image, k_size):
    import torch
    import torch.nn.functional as F
    if isinstance(image, torch.Tensor):
        return - F.max_pool2d(-image, k_size * 2 + 1, 1, k_size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (k_size, k_size))
        eroded = cv2.erode(image, kernel)
    return eroded


# 可获取图像中多个人脸的人脸点
class fdfa_multi(fdfa):
    def __call__(self, frame):
        rect_list = self.face_detect(frame)
        if rect_list is None:
            return None
        lmk_list = []

        for rect in rect_list:
            lmk = self.fa_run(frame, rect)
            if lmk is None:
                continue
            lmk = np.ascontiguousarray(lmk[:, :2])
            lmk_list.append(lmk)
        return lmk_list


class PreProcessPeople:
    def __init__(self):
        # 不要在子线程开启cuda
        self.fpm = fdfa_multi(True)    # 人脸点获取，用于test

    def resize(self, image, scale):
        if scale > 2:
            image = cv2.resize(image, (0, 0), fx=2, fy=2)
            return self.resize(image, scale / 2.)
        elif scale < 0.5:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            return self.resize(image, scale * 2.)
        else:
            return cv2.resize(image, (0, 0), fx=scale, fy=scale)

    def image_resize_640(self, image, fp):
        # 图像缩放为长边640，padding至 640 x 640
        background = np.zeros((640, 640, 3), dtype=np.uint8)
        background[:, :, :] = 127
        h, w = image.shape[:2]
        max_len = max(w, h)
        fx = 640 / max_len
        image = self.resize(image, fx)
        h, w = image.shape[:2]
        h_s = 320 - h // 2
        w_s = 320 - w // 2
        background[h_s:h_s + h, w_s:w_s + w] = image
        return background, [(f * fx) + (w_s, h_s) for f in fp]

    def mask_resize_640(self, image):
        # mask缩放为长边640
        h, w = image.shape[:2]
        self.org_image_size = (h, w)

        max_len = max(w, h)
        fx = 640 / max_len
        image = cv2.resize(image, None, fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
        h, w = image.shape[:2]
        h_s = 320 - h // 2
        w_s = 320 - w // 2
        background = self.my_pad(image, [h_s, 640 - h_s - h], [w_s, 640 - w_s - w])
        self.scale = fx
        self.xywh = [w_s, h_s, w, h]
        return background

    # 获取人脸点中心与半径
    def face_point_to_center_scale(self, fp):
        c = fp.mean(0, keepdims=True)
        diff = fp - c
        scale = (diff ** 2).mean(-1) ** 0.5
        return c[0].astype(np.int32), scale.mean()

    def to_mask(self, mask):
        if len(mask.shape) == 2:
            mask = mask[np.newaxis]
        return (mask / 255.).astype(np.float32)

    # 绘制人脸区域
    def draw_circle(self, image, face_point):
        h, w = image.shape[:2]
        face_mask = np.zeros((h//4, w//4))
        a_c, a_s = self.face_point_to_center_scale(face_point[:33] / 4)
        cv2.circle(face_mask, (a_c[0], a_c[1]), int(a_s * 2), 255, -1)
        face_mask = cv2.GaussianBlur(face_mask, (7, 7), 2)
        return cv2.resize(face_mask, (h, w))

    def __call__(self, image, mask, json_data=None):
        face_list = self.fpm(image)
        # 缩放图像与人脸点
        image, face_list = self.image_resize_640(image, face_list)
        # 缩放mask
        mask_list = cv2.merge([mask, mask, mask])
        mask_list = self.mask_resize_640(mask_list)
        mask, mask, mask = cv2.split(mask_list)

        # 绘制人脸热力图
        face_mask_list = [self.draw_circle(image, face) for face in face_list]
        mask = self.to_mask(mask)

        face_list = np.concatenate([self.to_mask(s) for s in face_mask_list], 0)

        return image, mask, face_list

    def rebuild(self, output_mask):
        print(output_mask.shape)
        x, y, w, h = self.xywh
        output_mask = output_mask[:, y:y+h, x:x+w]
        print(output_mask.shape)
        h, w = self.org_image_size
        output_mask = [cv2.resize(mask, (w, h)) for mask in output_mask]
        return output_mask

    # 补边
    def my_pad(self, image, pad_h=0, pad_w=0):
        if not isinstance(pad_h, (tuple, list)):
            pad_h = (pad_h, pad_h)
        if not isinstance(pad_w, (tuple, list)):
            pad_w = (pad_w, pad_w)
        pad_list = [pad_h, pad_w]
        if len(image.shape) == 3:
            pad_list = pad_list + [(0, 0)]

        image = np.pad(image, pad_list)
        return image


class Operation:
    def __init__(self):
        self.balance = None
        self.acne = None
        from lollipop.segment.Segment import Segment
        self.segment = Segment()
        self.load_module()
        self.lut = np.clip((np.arange(256) - 64) * 4 + 64, 0, 255).astype(np.uint8)
        self.pp = PreProcessPeople()

        hsv = np.ones((16, 1, 3), dtype=np.uint8) * 255
        for i in range(16):
            ix = i % 4
            iy = i // 4
            i = ix * 4 + iy
            hsv[i, :, 0] = 255 // 16 * i
        color_list = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]
        self.color_list = [tuple(c) for c in color_list] * 10

    def init_model(self):
        if self.balance is None:
            self.load_module()

    def get_tblr(self, contour, image_shape):
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
        t = y.min()
        b = y.max()
        l = x.min()
        r = x.max()
        img_h, img_w = image_shape[:2]
        h = b - t
        w = r - l
        pad_h = int(h * 0.15)
        pad_w = int(w * 0.15)
        t = max(0, t - pad_h)
        b = min(img_h - 1, b + pad_h)
        l = max(0, l - pad_w)
        r = min(img_w - 1, r + pad_w)
        return t, b, l, r

    def process(self, image_org, flag="not"):
        image = image_org.copy()
        mask = cv2.LUT(self.segment.get_body(image), self.lut)[..., np.newaxis]
        ret, thresh = cv2.threshold((mask).astype(np.uint8), 127, 255, 0)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        all_contour = np.concatenate(contours, 0)
        t, b, l, r = self.get_tblr(all_contour, image.shape)
        patch = image[t:b, l:r]
        with torch.no_grad():
            body_mask = self.segment.get_body(patch)
            image_patch, mask_patch, face_patch = self.pp(patch, body_mask)
            output = self.instance_segment(image_patch, face_patch, mask_patch)
            mask_list = self.pp.rebuild(output)

        for i, mask_one in enumerate(mask_list):
            mask_one = mask_one[..., np.newaxis] * 0.5
            patch = patch * (1 - mask_one) + mask_one * self.color_list[i]
        image[t:b, l:r] = np.clip(patch, 0, 255).astype(np.uint8)
        mask[t:b, l:r] = body_mask[..., np.newaxis]
        # cv2.imshow('mask', mask)
        
        if flag == 'ex':
            output = np.concatenate([np.concatenate([image] * 2, 0)] * 2, 1)
            output = cv2.resize(output, (0,0), fx=0.5, fy=0.5)
        else:
            output = image 
        return output

    def load_module(self):
        from Model.Res50.LightSolo import InstancePeopleSegment
        self.instance_segment = InstancePeopleSegment()


if __name__ == '__main__':
    from tqdm import tqdm
    from moviepy.editor import *
    import glob
    op = Operation()
    for f in glob.glob('/data/HumanTest/images/*.*'):
        image = cv2.imread(f)
        # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        img = op.process(image)
        # cv2.imshow('', img)
        cv2.waitKey()
        # cv2.imwrite('output_.png', img)
        # cv2.imshow('', img)

    # cv2.waitKey()

