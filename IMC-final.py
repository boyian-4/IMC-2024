import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--n', type=int)
arg('--t', type=float)
arg('--m', type=int)
arg('--p', type=int)
args = parser.parse_args()


# General utilities
import os
from tqdm import tqdm
from time import time
from fastprogress import progress_bar
import gc
import numpy as np
import pandas as pd
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
import concurrent.futures
from collections import Counter
import math

# CV/ML
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import kornia as K
import kornia.feature as KF
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import torchvision

# 3D reconstruction
import pycolmap

import glob
import matplotlib
from matplotlib import pyplot as plt

# dkm
import sys
sys.path.append('/kaggle/input/dkm-dependencies/DKM/')
from dkm.utils.utils import tensor_to_pil, get_tuple_transform_ops
from dkm import DKMv3_outdoor

# LoFTR
from kornia.feature import LoFTR

# LightGlue
from lightglue import match_pair
from lightglue import ALIKED, SuperPoint, DoGHardNet, LightGlue, DISK, SIFT
from lightglue.utils import load_image, rbd


# In[4]:


print('Kornia version', K.__version__)
print('Pycolmap version', pycolmap.__version__)


# # Configurations

# In[5]:


class CONFIG:
    # DEBUG Settings
    DRY_RUN = False
    DRY_RUN_MAX_IMAGES = 10

    # Pipeline settings
    NUM_CORES = 2
    
    # COLMAP Reconstruction
    CAMERA_MODEL = "simple-radial"
    
    # Rotation correction
    ROTATION_CORRECTION = False
    
    # Keypoints handling
    MERGE_PARAMS = {
        "min_matches" : 15,
        
        # When merging keypoints, it is enable to filtering matches with cv2.findFundamentalMatrix.
        "filter_FundamentalMatrix" : False,
        "filter_iterations" : 10,
        "filter_threshold" : 8,
    }
    
    # Keypoints Extraction
    use_aliked_lightglue = True
    use_doghardnet_lightglue = False
    use_superpoint_lightglue = False
    use_disk_lightglue = False
    use_sift_lightglue = False
    use_loftr = False
    use_dkm = False
    use_superglue = False
    use_matchformer = False
        
    # Keypoints Extraction Parameters
    params_aliked_lightglue = {
        "num_features" : args.n,
        "detection_threshold" : args.t,
        "min_matches" : args.m,
        "resize_to" : 1024,
    }
    
    params_doghardnet_lightglue = {
        "num_features" : 8192,
        "detection_threshold" : 0.001,
        "min_matches" : 15,
        "resize_to" : 1024,
    }
    
    params_superpoint_lightglue = {
        "num_features" : 4096,
        "detection_threshold" : 0.005,
        "min_matches" : 15,
        "resize_to" : 1024,
    }
    
    params_disk_lightglue = {
        "num_features" : 8192,
        "detection_threshold" : 0.001,
        "min_matches" : 15,
        "resize_to" : 1024,
    }

    params_sift_lightglue = {
        "num_features" : 8192,
        "detection_threshold" : 0.001,
        "min_matches" : 15,
        "resize_to" : 1024,
    }

    params_loftr = {
        "resize_small_edge_to" : 750,
        "min_matches" : 15,
    }
    
    params_dkm = {
        "num_features" : 2048,
        "detection_threshold" : 0.4,
        "min_matches" : 15,
        "resize_to" : (540, 720),    
    }
    
    # superpoint + superglue  ...  https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416873
    params_sg1 = {
        "sg_config" : 
        {
            "superpoint": {
                "nms_radius": 4, 
                "keypoint_threshold": 0.005,
                "max_keypoints": -1,
            },
            "superglue": {
                "weights": "outdoor",
                "sinkhorn_iterations": 20,
                "match_threshold": 0.2,
            },
        },
        "resize_to": 1088,
        "min_matches": 15,
    }
    params_sg2 = {
        "sg_config" : 
        {
            "superpoint": {
                "nms_radius": 4, 
                "keypoint_threshold": 0.005,
                "max_keypoints": -1,
            },
            "superglue": {
                "weights": "outdoor",
                "sinkhorn_iterations": 20,
                "match_threshold": 0.2,
            },
        },
        "resize_to": 1280,
        "min_matches": 15,
    }
    params_sg3 = {
        "sg_config" : 
        {
            "superpoint": {
                "nms_radius": 4, 
                "keypoint_threshold": 0.005,
                "max_keypoints": -1,
            },
            "superglue": {
                "weights": "outdoor",
                "sinkhorn_iterations": 20,
                "match_threshold": 0.2,
            },
        },
        "resize_to": 1376,
        "min_matches": 15,
    }
    params_sgs = [params_sg1, params_sg2, params_sg3]
    
    params_matchformer = {
        "detection_threshold" : 0.15,
        "resize_to" : (560, 750),
        "num_features" : 2000,
        "min_matches" : 15, 
    }


# In[6]:


device=torch.device('cuda')


# # COLMAP utilities

# In[7]:


# Code to manipulate a colmap database.
# Forked from https://github.com/colmap/colmap/blob/dev/scripts/python/database.py

# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.

import sys
import sqlite3
import numpy as np


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))


# # h5 to colmap db

# In[8]:


# Code to interface DISK with Colmap.
# Forked from https://github.com/cvlab-epfl/disk/blob/37f1f7e971cea3055bb5ccfc4cf28bfd643fa339/colmap/h5_to_db.py

#  Copyright [2020] [Michał Tyszkiewicz, Pascal Fua, Eduard Trulls]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os, argparse, h5py, warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ExifTags


def get_focal(image_path, err_on_default=False):
    image         = Image.open(image_path)
    max_size      = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size
    
    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal

def create_camera(db, image_path, camera_model):
    image         = Image.open(image_path)
    width, height = image.size

    focal = get_focal(image_path)

    if camera_model == 'simple-pinhole':
        model = 0 # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == 'pinhole':
        model = 1 # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == 'opencv':
        model = 4 # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
         
    return db.add_camera(model, width, height, param_arr)


def add_keypoints(db, h5_path, image_path, img_ext, camera_model, single_camera = True):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]

        fname_with_ext = filename# + img_ext
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')

        if camera_id is None or not single_camera:
            camera_id = create_camera(db, path, camera_model)
        image_id = db.add_image(fname_with_ext, camera_id)
        fname_to_id[filename] = image_id

        db.add_keypoints(image_id, keypoints)

    return fname_to_id

def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(os.path.join(h5_path, 'matches.h5'), 'r')
    
    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                    continue
            
                matches = group[key_2][()]
                db.add_matches(id_1, id_2, matches)

                added.add(pair_id)

                pbar.update(1)
                
def import_into_colmap(img_dir,
                       feature_dir ='.featureout',
                       database_path = 'colmap.db',
                       img_ext='.jpg'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, CONFIG.CAMERA_MODEL, single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )

    db.commit()
    return


# # Rotation detection

# In[9]:


from torchvision.io import read_image as T_read_image
from torchvision.io import ImageReadMode
from torchvision import transforms as T
from check_orientation.pre_trained_models import create_model

def convert_rot_k(index):
    if index == 0:
        return 0
    elif index == 1:
        return 3
    elif index == 2:
        return 2
    else:
        return 1

class CheckRotationDataset(Dataset):
    def __init__(self, files, transform=None):
        self.transform = transform
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        imgPath = self.files[idx]
        image = T_read_image(imgPath, mode=ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        return image

def get_CheckRotation_dataloader(images, batch_size=1):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = CheckRotationDataset(images, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )
    return dataloader

def exec_rotation_detection(img_files, device):
    model = create_model("swsl_resnext50_32x4d")
    model.eval().to(device);
    
    dataloader = get_CheckRotation_dataloader(img_files)
    
    rots = []
    for idx, image in enumerate(dataloader):
        image = image.to(torch.float32).to(device)
        with torch.no_grad():
            prediction = model(image).detach().cpu().numpy()
            detected_rot = prediction[0].argmax()
            rot_k = convert_rot_k(detected_rot)
            rots.append(rot_k)
            print(f"{os.path.basename(img_files[idx])} > rot_k={rot_k}")
    return rots


# # Image Pairs

# In[10]:


# We will use ViT global descriptor to get matching shortlists.
def get_global_desc(fnames, model,
                    device =  torch.device('cpu')):
    model = model.eval()
    model= model.to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    global_descs_convnext=[]
    for i, img_fname_full in tqdm(enumerate(fnames),total= len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        img = Image.open(img_fname_full).convert('RGB')
        timg = transform(img).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            desc = model.forward_features(timg.to(device)).mean(dim=(-1,2))#
            #print (desc.shape)
            desc = desc.view(1, -1)
            desc_norm = F.normalize(desc, dim=1, p=2)
        #print (desc_norm)
        global_descs_convnext.append(desc_norm.detach().cpu())
    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all.to(torch.float32)

def convert_1d_to_2d(idx, num_images):
    idx1 = idx // num_images
    idx2 = idx % num_images
    return (idx1, idx2)

def get_pairs_from_distancematrix(mat):
    pairs = [ convert_1d_to_2d(idx, mat.shape[0]) for idx in np.argsort(mat.flatten())]
    pairs = [ pair for pair in pairs if pair[0] < pair[1] ]
    return pairs

def get_img_pairs_exhaustive(img_fnames, model, device):
    #index_pairs = []
    #for i in range(len(img_fnames)):
    #    for j in range(i+1, len(img_fnames)):
    #        index_pairs.append((i,j))
    #return index_pairs
    descs = get_global_desc(img_fnames, model, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    matching_list = get_pairs_from_distancematrix(dm)
    return matching_list


def get_image_pairs_shortlist(fnames,
                              sim_th = 0.6, # should be strict
                              min_pairs = 20,
                              exhaustive_if_less = 20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)

    model = timm.create_model('tf_efficientnet_b7',
                              checkpoint_path='/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b7/1/tf_efficientnet_b7_ra-6c08e654.pth')
    model.eval()
    descs = get_global_desc(fnames, model, device=device)

    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames, model, device)
    
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total+=1
    matching_list = sorted(list(set(matching_list)))
    return matching_list


# # Keypoints: LightGlue series

# In[11]:


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

def convert_coord(r, w, h, rotk):
    if rotk == 0:
        return r
    elif rotk == 1:
        rx = w-1-r[:, 1]
        ry = r[:, 0]
        return torch.concat([rx[None], ry[None]], dim=0).T
    elif rotk == 2:
        rx = w-1-r[:, 0]
        ry = h-1-r[:, 1]
        return torch.concat([rx[None], ry[None]], dim=0).T
    elif rotk == 3:
        rx = r[:, 1]
        ry = h-1-r[:, 0]
        return torch.concat([rx[None], ry[None]], dim=0).T

def detect_common(img_fnames,
                  model_name,
                  rots,
                  file_keypoints,
                  feature_dir = '.featureout',
                  num_features = 4096,
                  resize_to = 1024,
                  detection_threshold = 0.01,
                  device=torch.device('cpu'),
                  min_matches=15,verbose=True
                 ):
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)

    #####################################################
    # Extract keypoints and descriptions
    #####################################################
    dict_model = {
        "aliked" : ALIKED,
        "superpoint" : SuperPoint,
        "doghardnet" : DoGHardNet,
        "disk" : DISK,
        "sift" : SIFT,
    }
    extractor_class = dict_model[model_name]
    
    dtype = torch.float32 # ALIKED has issues with float16
    extractor = extractor_class(
        max_num_keypoints=num_features, detection_threshold=detection_threshold, resize=resize_to
    ).eval().to(device, dtype)
        
    dict_kpts_cuda = {}
    dict_descs_cuda = {}
    for (img_path, rot_k) in zip(img_fnames, rots):
        img_fname = img_path.split('/')[-1]
        key = img_fname
        with torch.inference_mode():
            image0 = load_torch_image(img_path, device=device).to(dtype)
            h, w = image0.shape[2], image0.shape[3]
            image1 = torch.rot90(image0, rot_k, [2, 3])
            feats0 = extractor.extract(image1)  # auto-resize the image, disable with resize=None
            kpts = feats0['keypoints'].reshape(-1, 2).detach()
            descs = feats0['descriptors'].reshape(len(kpts), -1).detach()
            kpts = convert_coord(kpts, w, h, rot_k)
            dict_kpts_cuda[f"{key}"] = kpts
            dict_descs_cuda[f"{key}"] = descs
            print(f"{model_name} > rot_k={rot_k}, kpts.shape={kpts.shape}, descs.shape={descs.shape}")
    del extractor
    gc.collect()

    #####################################################
    # Matching keypoints
    #####################################################
    lg_matcher = KF.LightGlueMatcher(model_name, {"width_confidence": -1,
                                            "depth_confidence": -1,
                                             "mp": True if 'cuda' in str(device) else False}).eval().to(device)
    
    cnt_pairs = 0
    with h5py.File(file_keypoints, mode='w') as f_match:
        for pair_idx in tqdm(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            
            kp1 = dict_kpts_cuda[key1]
            kp2 = dict_kpts_cuda[key2]
            desc1 = dict_descs_cuda[key1]
            desc2 = dict_descs_cuda[key2]
            with torch.inference_mode():
                dists, idxs = lg_matcher(desc1,
                                     desc2,
                                     KF.laf_from_center_scale_ori(kp1[None]),
                                     KF.laf_from_center_scale_ori(kp2[None]))
            if len(idxs)  == 0:
                continue
            n_matches = len(idxs)
            kp1 = kp1[idxs[:,0], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
            kp2 = kp2[idxs[:,1], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=np.concatenate([kp1, kp2], axis=1))
                cnt_pairs+=1
                print (f'{model_name}> {key1}-{key2}: {n_matches} matches @ {cnt_pairs}th pair({model_name}+lightglue)')            
            else:
                print (f'{model_name}> {key1}-{key2}: {n_matches} matches --> skipped')
    del lg_matcher
    torch.cuda.empty_cache()
    gc.collect()
    return

def detect_lightglue_common(
    img_fnames, model_name, index_pairs, feature_dir, device, file_keypoints, rots,
    resize_to=1024,
    detection_threshold=0.01, 
    num_features=4096, 
    min_matches=15,
):
    t=time()
    detect_common(
        img_fnames, model_name, rots, file_keypoints, feature_dir, 
        resize_to=resize_to,
        num_features=num_features, 
        detection_threshold=detection_threshold, 
        device=device,
        min_matches=min_matches,
    )
    gc.collect()
    t=time() -t 
    print(f'Features matched in  {t:.4f} sec ({model_name}+LightGlue)')
    return t


# # Keypoints: SuperGlue

# In[12]:


import sys
sys.path.append("../input/super-glue-pretrained-network")
from models.matching import Matching
from models.superpoint import SuperPoint as SG_SuperPoint
from models.superglue import SuperGlue
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          process_resize, frame2tensor,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

from torch.nn import functional as torchF  # For resizing tensor

def sg_imread(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image

# Preprocess
def sg_read_image(image, device, resize):
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, [resize,])
    
    unit_shape = 8
    w_new = w_new // unit_shape * unit_shape
    h_new = h_new // unit_shape * unit_shape
    
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    image = cv2.resize(image.astype('float32'), (w_new, h_new))

    inp = frame2tensor(image, "cpu")
    return image, inp, scales, (h, w)

class SGDataset(Dataset):
    def __init__(self, img_fnames, resize_to, device):
        self.img_fnames = img_fnames
        self.resize_to = resize_to
        self.device = device
        
    def __len__(self):
        return len(self.img_fnames)
    
    def __getitem__(self, idx):
        fname = self.img_fnames[idx]
        im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        _, image, scale, ori_shape = sg_read_image(im, self.device, self.resize_to)
        return image, torch.tensor([idx]), torch.tensor(ori_shape)

def get_superglue_dataloader(img_fnames, resize_to, device, batch_size=1):
    dataset = SGDataset(img_fnames, resize_to, device)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )
    return dataloader

def detect_superglue(
    img_fnames, index_pairs, feature_dir, device, sg_config, file_keypoints, 
    resize_to=750, min_matches=15
):    
    t=time()

    fnames1, fnames2, idxs1, idxs2 = [], [], [], []
    for pair_idx in progress_bar(index_pairs):
        idx1, idx2 = pair_idx
        fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
        fnames1.append(fname1)
        fnames2.append(fname2)
        idxs1.append(idx1)
        idxs2.append(idx2)
        
    dataloader = get_superglue_dataloader( img_fnames, resize_to, device)

    #####################################################
    # Extract keypoints and descriptions
    #####################################################
    superpoint = SG_SuperPoint(sg_config["superpoint"]).eval().to(device)
    dict_features_cuda = {}
    dict_shapes = {}
    dict_images = {}
    for X in dataloader:
        image, idx, ori_shape = X
        image = image[0].to(device)
        fname = img_fnames[idx]
        key = fname.split('/')[-1]
        
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = superpoint({'image': image})
            dict_features_cuda[key] = pred
            dict_shapes[key] = ori_shape
            dict_images[key] = image.half()
    del superpoint
    gc.collect()
    
    #####################################################
    # Matching keypoints
    #####################################################
    superglue = SuperGlue(sg_config["superglue"]).eval().to(device)
    weights = sg_config["superglue"]["weights"]
    cnt_pairs = 0
    
    with h5py.File(file_keypoints, mode='w') as f_match:
        for idx, (fname1, fname2) in enumerate(zip(fnames1, fnames2)):
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]

            data = {"image0": dict_images[key1], "image1": dict_images[key2]}
            data = {**data, **{k+'0': v for k, v in dict_features_cuda[key1].items()}}
            data = {**data, **{k+'1': v for k, v in dict_features_cuda[key2].items()}}
            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = {**data, **superglue(data)}
                pred = {k: v[0].detach().cpu().numpy().copy() for k, v in pred.items()}
            mkpts1, mkpts2 = pred["keypoints0"], pred["keypoints1"]
            matches, conf = pred["matches0"], pred["matching_scores0"]

            valid = matches > -1
            mkpts1 = mkpts1[valid]
            mkpts2 = mkpts2[matches[valid]]
            mconf = conf[valid]

            ori_shape_1 = dict_shapes[key1][0].numpy()
            ori_shape_2 = dict_shapes[key2][0].numpy()
            
            # Scaling coords
            mkpts1[:,0] = mkpts1[:,0] * ori_shape_1[1] / dict_images[key1].shape[3]   # X
            mkpts1[:,1] = mkpts1[:,1] * ori_shape_1[0] / dict_images[key1].shape[2]   # Y
            mkpts2[:,0] = mkpts2[:,0] * ori_shape_2[1] / dict_images[key2].shape[3]   # X
            mkpts2[:,1] = mkpts2[:,1] * ori_shape_2[0] / dict_images[key2].shape[2]   # Y  
            
            n_matches = mconf.shape[0]
            
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=np.concatenate([mkpts1, mkpts2], axis=1).astype(np.float32))
                cnt_pairs+=1
                print (f'{key1}-{key2}: {n_matches} matches @ {cnt_pairs}th pair(superglue/{resize_to}/{weights})')            
            else:
                print (f'{key1}-{key2}: {n_matches} matches --> skipped')            

    del superglue
    del dict_features_cuda
    del dict_images
    torch.cuda.empty_cache()
    gc.collect()
    t=time() -t 
    print(f'Features matched in  {t:.4f} sec')
    return t


# # Keypoints: DKM

# In[13]:


# Making kornia local features loading w/o internet
class KeyNetAffNetHardNet(KF.LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

    .. image:: _static/img/keynet_affnet.jpg
    """

    def __init__(
        self,
        num_features: int = 5000,
        upright: bool = False,
        device = torch.device('cpu'),
        scale_laf: float = 1.0,
    ):
        ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
        if not upright:
            weights = torch.load('/kaggle/input/kornia-local-feature-weights/OriNet.pth')['state_dict']
            ori_module.angle_detector.load_state_dict(weights)
        detector = KF.KeyNetDetector(
            False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()
        ).to(device)
        kn_weights = torch.load('/kaggle/input/kornia-local-feature-weights/keynet_pytorch.pth')['state_dict']
        detector.model.load_state_dict(kn_weights)
        affnet_weights = torch.load('/kaggle/input/kornia-local-feature-weights/AffNet.pth')['state_dict']
        detector.aff.load_state_dict(affnet_weights)
        
        hardnet = KF.HardNet(False).eval()
        hn_weights = torch.load('/kaggle/input/kornia-local-feature-weights/HardNetLib.pth')['state_dict']
        hardnet.load_state_dict(hn_weights)
        descriptor = KF.LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)


# In[14]:


LOCAL_FEATURE = 'KeyNetAffNetHardNet'
def detect_features(img_fnames,
                    num_feats = 20000,
                    upright = False,
                    device=torch.device('cpu'),
                    feature_dir = '.featureout',
                    resize_small_edge_to = 1200):
    if LOCAL_FEATURE == 'DISK':
        # Load DISK from Kaggle models so it can run when the notebook is offline.
        disk = KF.DISK().to(device)
        pretrained_dict = torch.load('/kaggle/input/disk/pytorch/depth-supervision/1/loftr_outdoor.ckpt', map_location=device)
        disk.load_state_dict(pretrained_dict['extractor'])
        disk.eval()
    if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
        feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
    if LOCAL_FEATURE == 'GFTTAffNetHardNet':
        feature = GFTTAffNetHardNet(num_feats, upright, device).to(device).eval()
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
         h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in progress_bar(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                timg = load_torch_image(img_path, device=device)
                H, W = timg.shape[2:]
                if resize_small_edge_to is None:
                    timg_resized = timg
                else:
                    timg_resized = K.geometry.resize(timg, resize_small_edge_to, antialias=True)
                    print(f'Resized {timg.shape} to {timg_resized.shape} (resize_small_edge_to={resize_small_edge_to})')
                h, w = timg_resized.shape[2:]
                if LOCAL_FEATURE == 'DISK':
                    features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                    kps1, descs = features.keypoints, features.descriptors
                    
                    lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
                    lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
                if LOCAL_FEATURE == 'GFTTAffNetHardNet':
                    lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
                lafs[:,:,0,:] *= float(W) / float(w)
                lafs[:,:,1,:] *= float(H) / float(h)
                desc_dim = descs.shape[-1]
                kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                f_laf[key] = lafs.detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return

def match_features(img_fnames,
                   index_pairs,
                   file_keypoints,
                   feature_dir = '.featureout',
                   device=torch.device('cpu'),
                   min_matches=15, 
                   force_mutual = True,
                   matching_alg='adalam'
                  ):
    assert matching_alg in ['smnn', 'adalam']
    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
        h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(file_keypoints, mode='w') as f_match:

        for pair_idx in progress_bar(index_pairs):
                    idx1, idx2 = pair_idx
                    fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                    key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
                    lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
                    lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
                    desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
                    desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
                    kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
                    kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
                    if matching_alg == 'adalam':
                        img1, img2 = cv2.imread(fname1), cv2.imread(fname2)
                        hw1, hw2 = img1.shape[:2], img2.shape[:2]
                        adalam_config = KF.adalam.get_adalam_default_config()
                        #adalam_config['orientation_difference_threshold'] = None
                        #adalam_config['scale_rate_threshold'] = None
                        adalam_config['force_seed_mnn']= False
                        adalam_config['search_expansion'] = 16
                        adalam_config['ransac_iters'] = 128
                        adalam_config['device'] = device
                        dists, idxs = KF.match_adalam(desc1, desc2,
                                                      lafs1, lafs2, # Adalam takes into account also geometric information
                                                      hw1=hw1, hw2=hw2,
                                                      config=adalam_config) # Adalam also benefits from knowing image size
                    else:
                        dists, idxs = KF.match_smnn(desc1, desc2, 0.98)
                    if len(idxs)  == 0:
                        continue
                    # Force mutual nearest neighbors
                    if force_mutual:
                        first_indices = get_unique_idxs(idxs[:,1])
                        idxs = idxs[first_indices]
                        dists = dists[first_indices]
                    n_matches = len(idxs)
                    kp1 = kp1[idxs[:,0], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
                    kp2 = kp2[idxs[:,1], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
                    if False:
                        print (f'{key1}-{key2}: {n_matches} matches')
                    group  = f_match.require_group(key1)
                    if n_matches >= min_matches:
                         group.create_dataset(key2, data=np.concatenate([kp1, kp2], axis=1))
    return


# # Keypoints merger

# In[15]:


def get_unique_idxs(A, dim=0):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices

def get_keypoint_from_h5(fp, key1, key2):
    rc = -1
    try:
        kpts = np.array(fp[key1][key2])
        rc = 0
        return (rc, kpts)
    except:
        return (rc, None)

def get_keypoint_from_multi_h5(fps, key1, key2):
    list_mkpts = []
    for fp in fps:
        rc, mkpts = get_keypoint_from_h5(fp, key1, key2)
        if rc == 0:
            list_mkpts.append(mkpts)
    if len(list_mkpts) > 0:
        list_mkpts = np.concatenate(list_mkpts, axis=0)
    else:
        list_mkpts = None
    return list_mkpts

def matches_merger(
    img_fnames,
    index_pairs,
    files_keypoints,
    save_file,
    feature_dir = 'featureout',
    filter_FundamentalMatrix = False,
    filter_iterations = 10,
    filter_threshold = 8,
):
    # open h5 files
    fps = [ h5py.File(file, mode="r") for file in files_keypoints ]

    with h5py.File(save_file, mode='w') as f_match:
        counter = 0
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]

            # extract keypoints
            mkpts = get_keypoint_from_multi_h5(fps, key1, key2)
            if mkpts is None:
                print(f"skipped key1={key1}, key2={key2}")
                continue

            ori_size = mkpts.shape[0]
            if mkpts.shape[0] < CONFIG.MERGE_PARAMS["min_matches"]:
                continue
            
            if filter_FundamentalMatrix:
                store_inliers = { idx:0 for idx in range(mkpts.shape[0]) }
                idxs = np.array(range(mkpts.shape[0]))
                for iter in range(filter_iterations):
                    try:
                        Fm, inliers = cv2.findFundamentalMat(
                            mkpts[:,:2], mkpts[:,2:4], cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
                        if Fm is not None:
                            inliers = inliers > 0
                            inlier_idxs = idxs[inliers[:, 0]]
                            #print(inliers.shape, inlier_idxs[:5])
                            for idx in inlier_idxs:
                                store_inliers[idx] += 1
                    except:
                        print(f"Failed to cv2.findFundamentalMat. mkpts.shape={mkpts.shape}")
                inliers = np.array([ count for (idx, count) in store_inliers.items() ]) >= filter_threshold
                mkpts = mkpts[inliers]
                if mkpts.shape[0] < 15:
                    print(f"skipped key1={key1}, key2={key2}: mkpts.shape={mkpts.shape} after filtered.")
                    continue
                #print(f"filter_FundamentalMatrix: {len(store_inliers)} matches --> {mkpts.shape[0]} matches")
            
            
            print (f'{key1}-{key2}: {ori_size} --> {mkpts.shape[0]} matches')            
            # regist tmp file
            group  = f_match.require_group(key1)
            group.create_dataset(key2, data=mkpts)
            counter += 1
    print( f"Ensembled pairs : {counter} pairs" )
    for fp in fps:
        fp.close()

def keypoints_merger(
    img_fnames,
    index_pairs,
    files_keypoints,
    feature_dir = 'featureout',
    filter_FundamentalMatrix = False,
    filter_iterations = 10,
    filter_threshold = 8,
):
    save_file = f'{feature_dir}/merge_tmp.h5'
    os.system('rm -rf {}'.format(save_file))
    matches_merger(
        img_fnames,
        index_pairs,
        files_keypoints,
        save_file,
        feature_dir = feature_dir,
        filter_FundamentalMatrix = filter_FundamentalMatrix,
        filter_iterations = filter_iterations,
        filter_threshold = filter_threshold,
    )
        
    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts=defaultdict(int)
    with h5py.File(save_file, mode='r') as f_match:
        for k1 in f_match.keys():
            group  = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0]+=total_kpts[k1]
                current_match[:, 1]+=total_kpts[k2]
                total_kpts[k1]+=len(matches)
                total_kpts[k2]+=len(matches)
                match_indexes[k1][k2]=current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:,0] = unique_match_idxs[k1][m2[:,0]]
            m2[:,1] = unique_match_idxs[k2][m2[:,1]]
            mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
                                    unique_kpts[k2][  m2[:,1]],
                                   ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1
    
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
    return


# # Keypoints wrapper function

# In[16]:


def wrapper_keypoints(
    img_fnames, index_pairs, feature_dir, device, timings, rots
):
    #############################################################
    # get keypoints
    #############################################################
    files_keypoints = []
    detect_features(img_fnames, 
                                20000,
                                feature_dir=feature_dir,
                                upright=False,
                                device=device,
                                resize_small_edge_to=1200
                               )
    file_keypoints = f"{feature_dir}/matches_keynet.h5"
    match_features(img_fnames, index_pairs, file_keypoints, feature_dir=feature_dir,device=device)
    files_keypoints.append( file_keypoints )
    if CONFIG.use_superglue:
        for params_sg in CONFIG.params_sgs:
            resize_to = params_sg["resize_to"]
            file_keypoints = f"{feature_dir}/matches_superglue_{resize_to}pix.h5"
            os.system('rm -rf {}'.format(file_keypoints))
            t = detect_superglue(
                img_fnames, index_pairs, feature_dir, device, 
                params_sg["sg_config"], file_keypoints, 
                resize_to=params_sg["resize_to"], 
                min_matches=params_sg["min_matches"],
            )
            gc.collect()
            files_keypoints.append( file_keypoints )
            timings['feature_matching'].append(t)

    if CONFIG.use_aliked_lightglue:
        model_name = "aliked"
        file_keypoints = f'{feature_dir}/matches_lightglue_{model_name}.h5'
        t = detect_lightglue_common(
            img_fnames, model_name, index_pairs, feature_dir, device, file_keypoints, rots,
            resize_to=CONFIG.params_aliked_lightglue["resize_to"],
            detection_threshold=CONFIG.params_aliked_lightglue["detection_threshold"],
            num_features=CONFIG.params_aliked_lightglue["num_features"],
            min_matches=CONFIG.params_aliked_lightglue["min_matches"],
        )
        gc.collect()
        files_keypoints.append(file_keypoints)
        timings['feature_matching'].append(t)

    if CONFIG.use_doghardnet_lightglue:
        model_name = "doghardnet"
        file_keypoints = f'{feature_dir}/matches_lightglue_{model_name}.h5'
        t = detect_lightglue_common(
            img_fnames, model_name, index_pairs, feature_dir, device, file_keypoints, rots,
            resize_to=CONFIG.params_doghardnet_lightglue["resize_to"],
            detection_threshold=CONFIG.params_doghardnet_lightglue["detection_threshold"],
            num_features=CONFIG.params_doghardnet_lightglue["num_features"],
            min_matches=CONFIG.params_doghardnet_lightglue["min_matches"],
        )
        gc.collect()
        files_keypoints.append(file_keypoints)
        timings['feature_matching'].append(t)

    if CONFIG.use_superpoint_lightglue:
        model_name = "superpoint"
        file_keypoints = f'{feature_dir}/matches_lightglue_{model_name}.h5'
        t = detect_lightglue_common(
            img_fnames, model_name, index_pairs, feature_dir, device, file_keypoints, rots,
            resize_to=CONFIG.params_superpoint_lightglue["resize_to"],
            detection_threshold=CONFIG.params_superpoint_lightglue["detection_threshold"],
            num_features=CONFIG.params_superpoint_lightglue["num_features"],
            min_matches=CONFIG.params_superpoint_lightglue["min_matches"],
        )
        gc.collect()
        files_keypoints.append(file_keypoints)
        timings['feature_matching'].append(t)

    if CONFIG.use_disk_lightglue:
        model_name = "disk"
        file_keypoints = f'{feature_dir}/matches_lightglue_{model_name}.h5'
        t = detect_lightglue_common(
            img_fnames, model_name, index_pairs, feature_dir, device, file_keypoints, rots,
            resize_to=CONFIG.params_disk_lightglue["resize_to"],
            detection_threshold=CONFIG.params_disk_lightglue["detection_threshold"],
            num_features=CONFIG.params_disk_lightglue["num_features"],
            min_matches=CONFIG.params_disk_lightglue["min_matches"],
        )
        gc.collect()
        files_keypoints.append(file_keypoints)
        timings['feature_matching'].append(t)

    if CONFIG.use_sift_lightglue:
        model_name = "sift"
        file_keypoints = f'{feature_dir}/matches_lightglue_{model_name}.h5'
        t = detect_lightglue_common(
            img_fnames, model_name, index_pairs, feature_dir, device, file_keypoints, rots,
            resize_to=CONFIG.params_sift_lightglue["resize_to"],
            detection_threshold=CONFIG.params_sift_lightglue["detection_threshold"],
            num_features=CONFIG.params_sift_lightglue["num_features"],
            min_matches=CONFIG.params_sift_lightglue["min_matches"],
        )
        gc.collect()
        files_keypoints.append(file_keypoints)
        timings['feature_matching'].append(t)

    if CONFIG.use_loftr:
        file_keypoints = f'{feature_dir}/matches_loftr_{CONFIG.params_loftr["resize_small_edge_to"]}pix.h5'
        t = detect_loftr(
            img_fnames, index_pairs, feature_dir, device, file_keypoints,
            resize_small_edge_to=CONFIG.params_loftr["resize_small_edge_to"],
            min_matches=CONFIG.params_loftr["min_matches"],
        )
        gc.collect()
        files_keypoints.append( file_keypoints )
        timings['feature_matching'].append(t)

    if CONFIG.use_dkm:
        file_keypoints = f'{feature_dir}/matches_dkm.h5'
        t = detect_dkm(
            img_fnames, index_pairs, feature_dir, device, file_keypoints,
            resize_to=CONFIG.params_dkm["resize_to"], 
            detection_threshold=CONFIG.params_dkm["detection_threshold"], 
            num_features=CONFIG.params_dkm["num_features"], 
            min_matches=CONFIG.params_dkm["min_matches"]
        )
        gc.collect()
        files_keypoints.append(file_keypoints)
        timings['feature_matching'].append(t)

    if CONFIG.use_matchformer:
        file_keypoints = f'{feature_dir}/matches_matchformer_{CONFIG.params_matchformer["resize_to"]}pix.h5'
        t = detect_matchformer(
            img_fnames, index_pairs, feature_dir, device, file_keypoints,
            resize_to=CONFIG.params_matchformer["resize_to"],
            num_features=CONFIG.params_matchformer["num_features"], 
            min_matches=CONFIG.params_matchformer["min_matches"]
        )
        gc.collect()
        files_keypoints.append( file_keypoints )
        timings['feature_matching'].append(t)

    #############################################################
    # merge keypoints
    #############################################################
    keypoints_merger(
        img_fnames,
        index_pairs,
        files_keypoints,
        feature_dir = feature_dir,
        filter_FundamentalMatrix = CONFIG.MERGE_PARAMS["filter_FundamentalMatrix"],
        filter_iterations = CONFIG.MERGE_PARAMS["filter_iterations"],
        filter_threshold = CONFIG.MERGE_PARAMS["filter_threshold"],
    )    
    return timings


# # Reconstruction wrapper function

# In[17]:


def reconstruct_from_db(dataset, scene, feature_dir, img_dir, timings, image_paths):
    scene_result = {}
    #############################################################
    # regist keypoints from h5 into colmap db
    #############################################################
    database_path = f'{feature_dir}/colmap.db'
    if os.path.isfile(database_path):
        os.remove(database_path)
    gc.collect()
    import_into_colmap(img_dir, feature_dir=feature_dir, database_path=database_path)
    output_path = f'{feature_dir}/colmap_rec'

    #############################################################
    # Calculate fundamental matrix with colmap api
    #############################################################
    t=time()
    options = pycolmap.SiftMatchingOptions()
    
    pycolmap.match_exhaustive(database_path, sift_options=options)
    t=time() - t 
    timings['RANSAC'].append(t)
    print(f'RANSAC in  {t:.4f} sec')

    #############################################################
    # Execute bundle adjustmnet with colmap api
    # --> Bundle adjustment Calcs Camera matrix, R and t
    #############################################################
    t=time()
    # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
    mapper_options = pycolmap.IncrementalMapperOptions()
    mapper_options.min_model_size = 3
    os.makedirs(output_path, exist_ok=True)
    maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, output_path=output_path, options=mapper_options)
    print(maps)
    clear_output(wait=False)
    t=time() - t
    timings['Reconstruction'].append(t)
    print(f'Reconstruction done in  {t:.4f} sec')

    #############################################################
    # Extract R,t from maps 
    #############################################################            
    imgs_registered  = 0
    best_idx = None
    list_num_images = []            
    print ("Looking for the best reconstruction")
    if isinstance(maps, dict):
        for idx1, rec in maps.items():
            print (idx1, rec.summary())
            list_num_images.append( len(rec.images) )
            if len(rec.images) > imgs_registered:
                imgs_registered = len(rec.images)
                best_idx = idx1
    list_num_images = np.array(list_num_images)
    print(f"list_num_images = {list_num_images}")
    if best_idx is not None:
        print (maps[best_idx].summary())
        for k, im in maps[best_idx].images.items():
            key1 = f'test/{dataset}/images/{im.name}'
            scene_result[key1] = {}
            scene_result[key1]["R"] = deepcopy(im.rotmat())
            scene_result[key1]["t"] = deepcopy(np.array(im.tvec))

    print(f'Registered: {dataset} / {scene} -> {len(scene_result)} images')
    print(f'Total: {dataset} / {scene} -> {len(image_paths)} images')
    print(timings)
    return scene_result


# # Submission utilities

# In[18]:


def arr_to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])

# Function to create a submission file.
def create_submission(out_results, data_dict):
    with open(f'submission.csv', 'w') as f:
        f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')
        for dataset in data_dict:
            if dataset in out_results:
                res = out_results[dataset]
            else:
                res = {}
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R":{}, "t":{}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print (image)
                        R = scene_res[image]['R'].reshape(-1)
                        T = scene_res[image]['t'].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f'{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')


# # Main

# In[19]:


src = '/kaggle/input/image-matching-challenge-2024'

# Get data from csv.
data_dict = {}
with open(f'{src}/sample_submission.csv', 'r') as f:
    for i, l in enumerate(f):
        # Skip header.
        if l and i > 0:
            image, dataset, scene, _, _ = l.strip().split(',')
            if dataset not in data_dict:
                data_dict[dataset] = {}
            if scene not in data_dict[dataset]:
                data_dict[dataset][scene] = []
            data_dict[dataset][scene].append(image)
            
            if CONFIG.DRY_RUN:
                if len(data_dict[dataset][scene]) == CONFIG.DRY_RUN_MAX_IMAGES:
                    break
                    
for dataset in data_dict:
    for scene in data_dict[dataset]:
        print(f'{dataset} / {scene} -> {len(data_dict[dataset][scene])} images')


# In[20]:


out_results = {}
timings = {
    "rotation_detection" : [],
    "shortlisting":[],
   "feature_detection": [],
   "feature_matching":[],
   "RANSAC": [],
   "Reconstruction": []
}

gc.collect()
datasets = []
for dataset in data_dict:
    datasets.append(dataset)

with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG.NUM_CORES) as executors:
    futures = defaultdict(dict)
    for dataset in datasets:
        print(dataset)
        if dataset not in out_results:
            out_results[dataset] = {}
        for scene in data_dict[dataset]:
            print(scene)
            # Fail gently if the notebook has not been submitted and the test data is not populated.
            # You may want to run this on the training data in that case?
            img_dir = f'{src}/test/{dataset}/images'
            if not os.path.exists(img_dir):
                continue

            out_results[dataset][scene] = {}
            img_fnames = [f'{src}/{x}' for x in data_dict[dataset][scene]]
            print (f"Got {len(img_fnames)} images")
            feature_dir = f'featureout/{dataset}_{scene}'
            if not os.path.isdir(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)

            #############################################################
            # get image rotations
            #############################################################
            t = time()
            if CONFIG.ROTATION_CORRECTION:
                rots = exec_rotation_detection(img_fnames, device)
            else:
                rots = [ 0 for fname in img_fnames ]
            t = time()-t
            timings['rotation_detection'].append(t)
            print (f'rotation_detection for {len(img_fnames)} images : {t:.4f} sec')
            gc.collect()
            
            #############################################################
            # get image pairs
            #############################################################
            t=time()
            index_pairs = get_image_pairs_shortlist(img_fnames,
                                  sim_th = 1.0, # should be strict
                                  min_pairs = args.p, # we select at least min_pairs PER IMAGE with biggest similarity
                                  exhaustive_if_less = args.p,
                                  device=device)
            t=time() -t 
            timings['shortlisting'].append(t)
            print (f'{len(index_pairs)}, pairs to match, {t:.4f} sec')
            gc.collect()

            #############################################################
            # get keypoints
            #############################################################            
            keypoints_timings = wrapper_keypoints(
                img_fnames, index_pairs, feature_dir, device, timings, rots
            )
            timings['feature_matching'] = keypoints_timings['feature_matching']
            gc.collect()

            #############################################################
            # kick COLMAP reconstruction
            #############################################################            
            futures[dataset][scene] = executors.submit(
                reconstruct_from_db, 
                dataset, scene, feature_dir, img_dir, timings, data_dict[dataset][scene])
                
    #############################################################
    # reconstruction results
    #############################################################            
    for dataset in datasets:
        for scene in data_dict[dataset]:
            # wait to complete COLMAP reconstruction
            result = futures[dataset][scene].result()
            if result is not None:
                out_results[dataset][scene] = result   # get R and t from result
    
    create_submission(out_results, data_dict)
    gc.collect()


os.system("cat submission.csv")

import pandas as pd
data = pd.read_csv('submission.csv')

os.system('rm -rf ./*')

data.to_csv('submission.csv', index=False)
