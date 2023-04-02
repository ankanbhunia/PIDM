
import os
import cv2
import math
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F

class DemoDataset(object):
    def __init__(self, data_root, opt, load_from_dataset=False):
        super().__init__()    
        self.LIMBSEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        self.COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        self.img_size = tuple([int(item) for item in opt.sub_path.split('-')])
        self.data_root = data_root
        self.load_from_dataset = load_from_dataset # load from deepfashion dataset

    def load_item(self, reference_img_path, label_path=None):
        if self.load_from_dataset:
            reference_img_path = self.transfrom_2_real_path(reference_img_path)
            label_path = self.transfrom_2_real_path(label_path)
        else:
            reference_img_path = self.transfrom_2_demo_path(reference_img_path)
            label_path = self.transfrom_2_demo_path(label_path)

        label_path = self.img_to_label(label_path)
        reference_img = self.get_image_tensor(reference_img_path)[None,:]
        label, _ = self.get_label_tensor(label_path)
        label = label[None,:]

        return {'reference_image':reference_img, 'target_skeleton':label}
    
    def get_image_tensor(self, path):
        img = Image.open(path)
        img = F.resize(img, self.img_size)
        img = F.to_tensor(img)
        img = F.normalize(img, (0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        return img    

    def get_label_tensor(self, path, param={}):
        canvas = np.zeros((self.img_size[0], self.img_size[1], 3)).astype(np.uint8)
        keypoint = np.loadtxt(path)
        keypoint = self.trans_keypoins(keypoint, param, (self.img_size[0], self.img_size[1]))
        stickwidth = 4
        for i in range(18):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, self.COLORS[i], thickness=-1)
        joints = []
        for i in range(17):
            Y = keypoint[np.array(self.LIMBSEQ[i])-1, 0]
            X = keypoint[np.array(self.LIMBSEQ[i])-1, 1]            
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.COLORS[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)
        pose = F.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

        tensors_dist = 0
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
            im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
            tensor_dist = F.to_tensor(Image.fromarray(im_dist))
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
            e += 1

        label_tensor = torch.cat((pose, tensors_dist), dim=0)
        if int(keypoint[14, 0]) != -1 and int(keypoint[15, 0]) != -1:
            y0, x0 = keypoint[14, 0:2]
            y1, x1 = keypoint[15, 0:2]
            face_center = torch.tensor([y0, x0, y1, x1]).float()
        else:
            face_center = torch.tensor([-1, -1, -1, -1]).float()
        return label_tensor, face_center     

    def transfrom_2_demo_path(self, item):
        item = os.path.join(self.data_root ,item)
        return item

    def transfrom_2_real_path(self, item):
        # item, ext = os.path.splitext(item)
        if 'WOMEN' in item:
            item = os.path.basename(item)
            path = ['img/WOMEN']
            name = item.split('WOMEN')[-1]
        elif 'MEN' in item:
            item = os.path.basename(item)
            path = ['img/MEN']
            name = item.split('MEN')[-1]
        else:
            return item
        path.append(name.split('id0')[0])
        path.append('id_0'+ name.split('id0')[-1][:7])
        filename = name.split(name.split('id0')[-1][:7])[-1]
        count=0
        for i in filename.split('_')[-1]:
            try:
                int(i)
                count+=1
            except:
                pass
        filename = filename.split('_')[0]+'_' \
            +filename.split('_')[-1][:count] + '_' \
            +filename.split('_')[-1][count:]
        path.append(filename)
        path = os.path.join(*path)
        return os.path.join(self.data_root, path)

    def img_to_label(self, path):     
        return path.replace('img/', 'pose/').replace('.png', '.txt').replace('.jpg', '.txt')

    def trans_keypoins(self, keypoints, param, img_size):
        # find missing index
        missing_keypoint_index = keypoints == -1

        # crop the white line in the original dataset
        keypoints[:,0] = (keypoints[:,0]-40)

        # resize the dataset
        img_h, img_w = img_size
        scale_w = 1.0/176.0 * img_w
        scale_h = 1.0/256.0 * img_h

        if 'scale_size' in param and param['scale_size'] is not None:
            new_h, new_w = param['scale_size']
            scale_w = scale_w / img_w * new_w
            scale_h = scale_h / img_h * new_h
        

        if 'crop_param' in param and param['crop_param'] is not None:
            w, h, _, _ = param['crop_param']
        else:
            w, h = 0, 0

        keypoints[:,0] = keypoints[:,0]*scale_w - w
        keypoints[:,1] = keypoints[:,1]*scale_h - h
        keypoints[missing_keypoint_index] = -1
        return keypoints


