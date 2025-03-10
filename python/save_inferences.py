import sklearn.metrics
import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import numpy as np
import open3d as o3d
import os
import sys
import cv2
# add the path to the folder where the tenext.py file is located
# sys.path.append(os.path.abspath("/home/antonio/Antonio/virtual_envs/trav_analysis/scripts/TeNeXt/"))
# from MinkUtrav.minkUnet_custom import MinkUtravA, MinkUNet14A
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from plyfile import PlyData
from datasets import preprocessingDataset
import pandas as pd
from plyfile import PlyData, PlyElement

def compute_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(6))
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.45, max_nn=30))
    pcd.orient_normals_to_align_with_direction()
    normals = np.asarray(pcd.normals)
    ey = o3d.geometry.PointCloud()
    ey.points = o3d.utility.Vector3dVector(pcd.points)
    ey.normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([ey])
    return ey


def inference(test_data, model, device, th, output_dir, cont):
    names = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label','u4')]
    color = []
    pcd_raw = o3d.io.read_point_cloud(test_data)
    points = np.asarray(pcd_raw.points)

    # Preprocess distance points
    # distances = np.linalg.norm(points, axis=1)
    # umbral_dist = np.where((distances < 1) | (distances >= 45))
    # umbral_dist = np.where((points[:,0] < -1))
    # new_points = np.delete(points, umbral_dist[0], axis=0)
    # pcd_rec = o3d.geometry.PointCloud()
    # pcd_rec.points = o3d.utility.Vector3dVector(points)

    coords_rec = points
    coords = coords_rec/ 0.2 # 0.2 is the voxel size
    features = np.ones((coords.shape[0], 1))

    coords_batched = ME.utils.batched_coordinates([coords])
    test_in_field = ME.TensorField(torch.from_numpy(features).to(dtype=torch.float32),coordinates=(coords_batched),
                        quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
                        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device)


    test_output = model(test_in_field.sparse())
    logit = test_output.slice(test_in_field)
    pred_raw = logit.F.detach().cpu().numpy()
    pred = np.where(pred_raw > th, 1, 0)
    data = np.concatenate((points,pred.reshape(pred.shape[0],1)), axis=1)
    # print(data.shape)
    hola = list(map(tuple, data))
    cloud_ply = np.array(hola, dtype=names)
    cloud_ply = PlyElement.describe(cloud_ply, 'vertex')
    # print("/media/arvc/Extreme SSD/seq9/inference" + test_data[-11:])
    PlyData([cloud_ply], byte_order='>').write(output_dir + test_data[-11:])



if __name__ == '__main__':

    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    th = 0.48
    output_dir = ""
    model2evaluate = ""
    dataset2evaluate = ""
    files=glob.glob(dataset2evaluate+"/*.ply")
    model = MinkUtravA(1, 1).to(device)
    # model.load_state_dict(torch.load(model2evaluate,map_location='cuda:0'))
    model.load_state_dict(torch.load(l))
    idx=1
    for cont, file in enumerate(tqdm(files)):
        idx=idx+1
        print(file)
        inference(files[idx], model, device, th, output_dir, cont)


