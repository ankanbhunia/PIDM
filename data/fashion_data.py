import os
import cv2 
import math
import lmdb
import numpy as np
from io import BytesIO
from PIL import Image

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from data.fashion_base_function import get_random_params, get_transform

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

class Dataset(Dataset):
    def __init__(self, opt, is_inference, labels_required = False):
        self.root = opt.path
        self.semantic_path = self.root
        path = os.path.join(self.root, str(opt.sub_path))
        self.path = path
        self.labels_required = labels_required

        # self.env = lmdb.open(
        #     path,
        #     max_readers=32,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False,
        # )

        # if not self.env:
        #     raise IOError('Cannot open lmdb dataset', path)

        self.file_path = 'train_pairs.txt' if not is_inference else 'test_pairs.txt'
        self.data = self.get_paths(self.root, self.file_path)
        self.is_inference = is_inference
        self.preprocess_mode = opt.preprocess_mode
        self.scale_param = opt.scale_param if not is_inference else 0
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def __len__(self):
        return len(self.data)


    def get_paths(self, root, path):
        fd = open(os.path.join(root, path))
        lines = fd.readlines()
        fd.close()

        image_paths = []
        for item in lines:
            dict_item={}
            item = item.strip().split(',')
            dict_item['source_image'] = [path.replace('.jpg', '.png') for path in item[1:]]
            dict_item['source_label'] = [os.path.join(self.semantic_path, self.img_to_label(path)) for path in dict_item['source_image']]
            dict_item['target_image'] = item[0].replace('.jpg', '.png')
            dict_item['target_label'] = os.path.join(self.semantic_path, self.img_to_label(dict_item['target_image']))
            image_paths.append(dict_item)
        return image_paths

    def open_lmdb(self):

        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(buffers=True)

    def __getitem__(self, index):

        if not hasattr(self, 'txn'):
            self.open_lmdb()


        path_item = self.data[index]
        i = np.random.choice(list(range(0, len(path_item['source_image']))))
        source_image_path = path_item['source_image'][i]
        source_label_path = path_item['source_label'][i]

        target_image_tensor, param = self.get_image_tensor(path_item['target_image'])

        if self.labels_required:
            target_label_tensor, target_face_center = self.get_label_tensor(path_item['target_label'], target_image_tensor, param)
        
        ref_tensor, param = self.get_image_tensor(source_image_path)

        if self.labels_required:
            label_ref_tensor, ref_face_center = self.get_label_tensor(source_label_path, ref_tensor, param)
        
        image_path = self.get_image_path(source_image_path, path_item['target_image'])

        if not self.is_inference:

            if torch.rand(1) < 0.5:

                target_image_tensor = F.hflip(target_image_tensor)
                ref_tensor = F.hflip(ref_tensor)

                if self.labels_required:

                    target_label_tensor = F.hflip(target_label_tensor)
                    label_ref_tensor = F.hflip(label_ref_tensor)


        if self.labels_required:

            input_dict = {'target_skeleton': target_label_tensor,
                        'target_image': target_image_tensor,
                        'target_face_center': target_face_center,

                        'source_image': ref_tensor,
                        'source_skeleton': label_ref_tensor,
                        'source_face_center': ref_face_center,

                        'path': image_path,
                        }

        else:

            input_dict = {'target_image': target_image_tensor,
                          'source_image': ref_tensor,
                         }

        return input_dict

    def get_image_path(self, source_name, target_name):
        source_name = self.path_to_fashion_name(source_name)
        target_name = self.path_to_fashion_name(target_name)
        image_path = os.path.splitext(source_name)[0] + '_2_' + os.path.splitext(target_name)[0]+'_vis.png'
        return image_path

    def path_to_fashion_name(self, path_in):
        path_in = path_in.split('img/')[-1]
        path_in = os.path.join('fashion', path_in)
        path_names = path_in.split('/') 
        path_names[3] = path_names[3].replace('_', '')
        path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
        path_names = "".join(path_names)
        return path_names

    def img_to_label(self, path):
        return path.replace('img/', 'pose/').replace('.png', '.txt')

    def get_image_tensor(self, path):
        with self.env.begin(write=False) as txn:
            key = f'{path}'.encode('utf-8')
            img_bytes = txn.get(key) 
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        param = get_random_params(img.size, self.scale_param)
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param

    def get_label_tensor(self, path, img, param):
        canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)
        keypoint = np.loadtxt(path)
        keypoint = self.trans_keypoins(keypoint, param, img.shape[1:])
        stickwidth = 4
        for i in range(18):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
        joints = []
        for i in range(17):
            Y = keypoint[np.array(self.limbSeq[i])-1, 0]
            X = keypoint[np.array(self.limbSeq[i])-1, 1]            
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
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


    def trans_keypoins(self, keypoints, param, img_size):
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



class Dataset_guide(Dataset):
    def __init__(self, opt, is_inference, labels_required = False):
        self.root = opt.path
        self.semantic_path = self.root
        path = os.path.join(self.root, str(opt.sub_path))
        self.path = path
        self.labels_required = labels_required
        

        # self.env = lmdb.open(
        #     path,
        #     max_readers=32,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False,
        # )

        # if not self.env:
        #     raise IOError('Cannot open lmdb dataset', path)

        self.file_path = 'train_pairs.txt' if not is_inference else 'test_pairs.txt'
        self.data = self.get_paths(self.root, self.file_path)
        self.len_ = len(self.data)
        self.is_inference = is_inference
        self.preprocess_mode = opt.preprocess_mode
        self.scale_param = opt.scale_param if not is_inference else 0
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def __len__(self):
        return len(self.data)


    def get_paths(self, root, path):
        fd = open(os.path.join(root, path))
        lines = fd.readlines()
        fd.close()

        image_paths = []
        for item in lines:
            dict_item={}
            item = item.strip().split(',')
            dict_item['source_image'] = [path.replace('.jpg', '.png') for path in item[1:]]
            dict_item['source_label'] = [os.path.join(self.semantic_path, self.img_to_label(path)) for path in dict_item['source_image']]
            dict_item['target_image'] = item[0].replace('.jpg', '.png')
            dict_item['target_label'] = os.path.join(self.semantic_path, self.img_to_label(dict_item['target_image']))
            image_paths.append(dict_item)
        return image_paths

    def open_lmdb(self):

        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(buffers=True)

    def __getitem__(self, index):

        if not hasattr(self, 'txn'):
            self.open_lmdb()

        path_item = self.data[index]

        guide_labels = (torch.rand(2) < 0.5)


        if guide_labels[0]:
            path_item_ref = path_item
        else:
            path_item_ref = self.data[np.random.choice(list(set(list(np.arange(self.len_)))-{index}))]

        if guide_labels[1]:
            path_item_pose = path_item
        else:
            path_item_pose = self.data[np.random.choice(list(set(list(np.arange(self.len_)))-{index}))]     


        i = np.random.choice(list(range(0, len(path_item_ref['source_image']))))

        ref_image_path = path_item_ref['source_image'][i]
        target_image_path = path_item['target_image']
        pose_label_path = path_item_pose['target_label']

        ref_image_tensor, param = self.get_image_tensor(ref_image_path)
        target_image_tensor, param = self.get_image_tensor(target_image_path)
        target_label_tensor, target_face_center = self.get_label_tensor(pose_label_path, target_image_tensor, param)

        if torch.rand(1) < 0.5:

            ref_image_tensor = F.hflip(ref_image_tensor)
            target_image_tensor = F.hflip(target_image_tensor)
            target_label_tensor = F.hflip(target_label_tensor)

        
        return {'pose_skeleton': target_label_tensor,
                'target_image': target_image_tensor,
                'ref_image_tensor': ref_image_tensor,
                'guide_label' : bin2dec(guide_labels, 2)}

    def get_image_path(self, source_name, target_name):
        source_name = self.path_to_fashion_name(source_name)
        target_name = self.path_to_fashion_name(target_name)
        image_path = os.path.splitext(source_name)[0] + '_2_' + os.path.splitext(target_name)[0]+'_vis.png'
        return image_path

    def path_to_fashion_name(self, path_in):
        path_in = path_in.split('img/')[-1]
        path_in = os.path.join('fashion', path_in)
        path_names = path_in.split('/') 
        path_names[3] = path_names[3].replace('_', '')
        path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
        path_names = "".join(path_names)
        return path_names

    def img_to_label(self, path):
        return path.replace('img/', 'pose/').replace('.png', '.txt')

    def get_image_tensor(self, path):
        with self.env.begin(write=False) as txn:
            key = f'{path}'.encode('utf-8')
            img_bytes = txn.get(key) 
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        param = get_random_params(img.size, self.scale_param)
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param

    def get_label_tensor(self, path, img, param):
        canvas = np.zeros((img.shape[1], img.shape[2], 3)).astype(np.uint8)
        keypoint = np.loadtxt(path)
        keypoint = self.trans_keypoins(keypoint, param, img.shape[1:])
        stickwidth = 4
        for i in range(18):
            x, y = keypoint[i, 0:2]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
        joints = []
        for i in range(17):
            Y = keypoint[np.array(self.limbSeq[i])-1, 0]
            X = keypoint[np.array(self.limbSeq[i])-1, 1]            
            cur_canvas = canvas.copy()
            if -1 in Y or -1 in X:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
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


    def trans_keypoins(self, keypoints, param, img_size):
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



