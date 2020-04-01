import torch
import torch.utils.data
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from skimage import io, transform
from torchvision import transforms

num_keypoints = 14 #for trousers


def gaussian_map(image_w, image_h, center_x, center_y, R):
    Gauss_map = np.zeros((image_h, image_w))
    mask_x = np.matlib.repmat(center_x, image_h, image_w)
    mask_y = np.matlib.repmat(center_y, image_h, image_w)
    x1 = np.arange(image_w)
    x_map = np.matlib.repmat(x1, image_h, 1)
    y1 = np.arange(image_h)
    y_map = np.matlib.repmat(y1, image_w, 1)
    y_map = np.transpose(y_map)
    Gauss_map = np.sqrt((x_map - mask_x)**2 + (y_map - mask_y)**2)
    Gauss_map = np.exp(-0.5 * Gauss_map / R)
    return Gauss_map


def gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R):
    ret = []
    # If landmark is located out of the image, values are 0.
    for i in range(len(landmark_in_pic)): 
        if landmark_in_pic[i] == 0:
            ret.append(np.zeros((image_w, image_h)))
        else:
            channel_map = gaussian_map(image_w, image_h, landmark_pos[i][0], landmark_pos[i][1], R)
            ret.append(channel_map.reshape((image_w, image_h)))
    return np.stack(ret, axis=0).astype(np.float32)

class BBoxCrop(object):

    def __call__(self, image, landmarks, x_1, y_1, x_2, y_2):
        h, w = image.shape[:2]
        
        top = y_1
        left = x_1
        # new_h = y_2 - y_1
        # new_w = x_2 - x_1
        new_w= x_2
        new_h= y_2

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return image, landmarks

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant')

        landmarks = landmarks * [new_w / w, new_h / h]

        return img, landmarks

class LandmarksNormalize(object):

    def __call__(self, image, landmark_pos):
        h, w = image.shape[:2]
        landmark_pos = landmark_pos / [float(w), float(h)]
        return landmark_pos


class LandmarksUnNormalize(object):

    def __call__(self, image, landmark_pos):
        h, w = image.shape[:2]
        landmark_pos = landmark_pos * [float(w), float(h)]
        return landmark_pos

class CheckLandmarks(object):

    def __call__(self, image, landmark_vis, landmark_in_pic, landmark_pos):
        h, w = image.shape[:2]
        landmark_vis = landmark_vis.copy()
        landmark_in_pic = landmark_in_pic.copy()
        landmark_pos = landmark_pos.copy()
        for i, vis in enumerate(landmark_vis):
            if (landmark_pos[i, 0] < 0) or (landmark_pos[i, 0] >= w) or (landmark_pos[i, 1] < 0) or (landmark_pos[i, 1] >= h):
                landmark_vis[i] = 0
                landmark_in_pic[i] = 0
        for i, in_pic in enumerate(landmark_in_pic):
            if in_pic == 0:
                landmark_pos[i, :] = 0
        return landmark_vis, landmark_in_pic, landmark_pos

class DeepFashionDataset(torch.utils.data.Dataset):
    def __init__(self, ds, root):
        
        self.ds = ds
        self.root = root
        self.to_tensor = transforms.ToTensor()
        self.bbox_crop = BBoxCrop()
        self.rescale224square = Rescale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.landmarks_normalize = LandmarksNormalize()
        self.landmarks_unnormalize = LandmarksUnNormalize()
        self.check_landmarks = CheckLandmarks()

    def __getitem__(self, i):
        
        sample = self.ds[i]
        image = io.imread(self.root + sample['file_name'])
        org_image_size= image.shape
        
        
        landmark_vis = np.array(list(map(int,self.ds[i]['keypoints'][2::3])))
        
        landmark_in_pic = [1] * num_keypoints #num of keypoints
        for j in range(len(landmark_vis)):
            if landmark_vis[j]==2:
                landmark_in_pic[j]=1
            elif landmark_vis[j]==1:
                landmark_in_pic[j]=1
            else:
                landmark_in_pic[j]=0

        
        landmark_pos_x = np.array(list(map(int,self.ds[i]['keypoints'][0::3]))).reshape(-1,1)
        landmark_pos_y = np.array(list(map(int,self.ds[i]['keypoints'][1::3]))).reshape(-1,1)
        landmark_pos = np.concatenate([landmark_pos_x, landmark_pos_y],axis=1)
        
        bbox = self.ds[i]['bbox']
        
        image, landmark_pos = self.bbox_crop(image, landmark_pos, bbox[0], bbox[1], bbox[2], bbox[3])
        image, landmark_pos = self.rescale224square(image, landmark_pos)
        
        landmark_vis, landmark_in_pic, landmark_pos = self.check_landmarks(image, landmark_vis, landmark_in_pic, landmark_pos)
        landmark_pos = landmark_pos.astype(np.float32)
        landmark_pos_normalized = self.landmarks_normalize(image, landmark_pos).astype(np.float32)
        
        image = image.copy()
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = image.float()

        ret = {}

        ret['image'] = image
        ret['landmark_vis'] = landmark_vis
        ret['landmark_in_pic'] = np.array(landmark_in_pic)
        ret['landmark_pos'] = landmark_pos
        ret['landmark_pos_normalized'] = landmark_pos_normalized
        image_h, image_w = image.size()[1:]
        R = num_keypoints
        ret['landmark_map'] = gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R)
        ret['landmark_map224'] = gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R)
        ret['org_image_size'] = np.array(org_image_size)
        ret['cropped_image_size'] = np.array([image_h, image_w])
        
        return ret

    def __len__(self):
        return len(self.ds)
