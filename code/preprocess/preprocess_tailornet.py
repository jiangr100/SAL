import os
import pickle
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import argparse

import trimesh
from scipy.spatial import cKDTree

'''
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
'''
# need to run it under the cgal path

import open3d as o3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=0.2)

    opt = parser.parse_args()

    data_splits = ['train_train', 'test_test']

    for split in data_splits:
        print('split: ', split)

        pkl_path = '/media/Data/data/tailornet_dataset_pkl/%s/old-t-shirt_female' % split
        idle_dir = '/media/Data/data/tailornet_dataset_pkl/idle/old-t-shirt_female'
        save_path = '/media/Data/data/tailornet_dataset_pkl/SAL_preprocessed/%s' % split

        for pkl in os.listdir(pkl_path):
            if not pkl.endswith('.pkl'):
                continue

            print('start processing ', pkl)
            start_time = time.time()

            with open(str(Path(pkl_path + '/' + pkl)), "rb") as f:
                data = pickle.load(f, encoding='latin1')

            with open(str(Path(idle_dir + '/' + data['idle_idx'] + '.pkl')), "rb") as f:
                idle_data = pickle.load(f, encoding='latin1')

            center = np.mean(data['o_verts'], axis=0)
            scale = 1
            pnts = (data['o_verts'] - center) / scale

            sigmas = []
            ptree = cKDTree(pnts)
            for p in np.array_split(pnts, 100, axis=0):
                d = ptree.query(p, 51)
                sigmas.append(d[0][:, -1])

            sigmas = np.concatenate(sigmas)
            sigmas_big = opt.sigma * np.ones_like(sigmas)

            sample = np.concatenate([pnts + np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                     pnts + np.expand_dims(sigmas_big, -1) * np.random.normal(0.0, 1.0,
                                                                                              size=pnts.shape)], axis=0)

            mesh = trimesh.Trimesh(
                vertices=pnts,
                faces=data['o_faces']
            )

            o3d_pc = o3d.geometry.PointCloud()
            o3d_pc.points = o3d.utility.Vector3dVector(pnts)
            query_points = o3d.geometry.PointCloud()
            query_points.points = o3d.utility.Vector3dVector(sample)
            dists_o3d = np.asarray(query_points.compute_point_cloud_distance(o3d_pc))

            '''
            start_time = time.time()
            _, dists_trimesh, _ = trimesh.proximity.closest_point(mesh, sample)
            dists_trimesh = np.array(dists_trimesh)

            print('trimesh time elasped ', time.time() - start_time)
            start_time = time.time()

            triangles = []
            for tri in mesh.triangles:
                a = Point_3((tri[0][0] - center[0]) / scale, (tri[0][1] - center[1]) / scale,
                            (tri[0][2] - center[2]) / scale)
                b = Point_3((tri[1][0] - center[0]) / scale, (tri[1][1] - center[1]) / scale,
                            (tri[1][2] - center[2]) / scale)
                c = Point_3((tri[2][0] - center[0]) / scale, (tri[2][1] - center[1]) / scale,
                            (tri[2][2] - center[2]) / scale)
                triangles.append(Triangle_3(a, b, c))
            tree = AABB_tree_Triangle_3_soup(triangles)

            dists = []
            for np_query in sample:
                cgal_query = Point_3(np_query[0].astype(np.double), np_query[1].astype(np.double),
                                     np_query[2].astype(np.double))

                cp = tree.closest_point(cgal_query)
                cp = np.array([cp.x(), cp.y(), cp.z()])
                dist = np.sqrt(((cp - np_query) ** 2).sum(axis=0))

                dists.append(dist)
            dists = np.array(dists)

            print(max(dists - dists_trimesh))
            print('time elapsed CGAL ', time.time() - start_time)
            exit(0)
            '''

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            file_name = pkl.split('.')[0]
            np.save(str(Path(save_path + '/' + file_name + '.npy', pnts)))
            np.save(str(Path(save_path + '/' + file_name + '_dist_triangle.npy')),
                    np.concatenate([sample, np.expand_dims(dists_o3d, axis=-1)], axis=-1))
            np.save(str(Path(save_path + '/' + file_name + '_normalization.npy')),
                    {"center": center, "scale": scale})

            print('done processing %s, time elapsed %f' % (pkl, time.time() - start_time))






