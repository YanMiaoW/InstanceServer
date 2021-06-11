from torch import nn
import torch.nn.functional as F
import torch
import glob
import numpy as np
import os
import warnings
# warnings.filterwarnings("ignore")


import sys
sys.path.append('/root/ym/code-python/InstanceSeg')
from LightSolo import Solo

class InstancePeopleSegment:
    def __init__(self):
        self.model = Solo(False)
        dir_path = Solo.pretrain_model_dir
        path_list = sorted(glob.glob(os.path.join(dir_path, '*.pth')))
        path_list = [os.path.join(dir_path,f'epoch_{Solo.model_epoch}_net_0.pth')]
        print('Load eval model:\n \t%s' % (path_list[-1]))
        self.model.load_state_dict(torch.load(path_list[-1], map_location='cpu'))
        # model_path = dir_path + '/'+'epoch_3_net_0.pth'
        # print('Load eval model:\n \t%s' % (model_path))
        # self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.model.eval()
        self.model.cuda()


    def image2tensor(self, image):
        mini_tensor = torch.from_numpy(image[:, :, ::-1].copy().transpose(2, 0, 1))[None].float() / 127.5 - 1
        mini_tensor = mini_tensor.cuda()
        print(mini_tensor.device)
        return mini_tensor

    def tensor2image(self, tensor):
        out = tensor.cpu().data[0].numpy().transpose(1, 2, 0)
        rgb = np.clip(out * 127.5 + 127.5, 0, 255).astype(np.uint8)
        bgr = rgb[:, :, ::-1]
        return bgr

    def __call__(self, image_org, face_mask, body_mask):
        image_tensor = self.image2tensor(image_org)
        face_mask = torch.from_numpy(face_mask)[None].float().cuda()
        body_mask = torch.from_numpy(body_mask)[None].float().cuda()


        with torch.no_grad():
            output_mask = self.model(image_tensor, face_mask, body_mask)

        return output_mask.cpu().data[0].numpy()


if __name__ == '__main__':
    m = Solo(True)
    image = torch.randn(3, 3, 640, 640)
    out_mask = m(image, image[:, :2], image[:, :1])

    print(out_mask.size())
