#!/usr/bin/env python3

import pcl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
import random
import open3d


np.set_printoptions(precision=3, suppress=True)

#------------ Load the pcd file-----------------
vel_msg = np.asarray(pcl.load(r'/home/harsh/catkin_ws/src/perception_pcl/pcl_ros/tools/376528000.pcd'))


n_segments = 100
n_bins = 300
r_max = 400
r_min = 0

segment_step = 2 * np.pi /n_segments
bin_step = (r_max - r_min) / n_bins

def project_5D(point3D):
        '''
        Args:
            point3D: shapes (n_row, 3), while 3 represent x,y,z axis in order.
        Returns:
            point5D: shapes (n_row, 3+2), while 5 represent x,y,z,seg,bin axis in order.
        '''
        x = point3D[:, 0]
        y = point3D[:, 1]
        z = point3D[:, 2]

        # index mapping
        angle = np.arctan2(y, x)
        segment_index = np.int32(np.floor((angle + np.pi) / segment_step))  # segment

        radius = np.sqrt(x ** 2 + y ** 2)
        bin_index = np.int32(np.floor((radius - r_min) / bin_step))  # bin

        point5D = np.vstack([point3D.T, segment_index, bin_index]).T

        return point5D


def filter_out_range(point5D):
        '''
        Args:
            point5D: shapes (n_row, 3+2), while 5 represent x,y,z,seg,bin axis in order.
        Returns:
            point5D: shapes (n_row_filtered, 5), while 5 represent x,y,z,seg,bin axis in order.
        '''
        radius = point5D[:, 4]  # [x,y,z,seg,bin]
        condition = np.logical_and(radius < r_max, radius > r_min)
        point5D = point5D[condition]

        return point5D

#-----------------converting 3D points into 5D points by adding their segment and bin info-------------------
point5D = project_5D(vel_msg)
point5D = filter_out_range(point5D)
point5D = point5D[np.argsort(point5D[:, 3])]
seg_list = np.int16(np.unique(point5D[:, 3]))
bin_list = np.int16(np.unique(point5D[:, 4]))


#---------------------getting the point with minimum z-coordinate (1 point per bin)----------------------------
min_z_points_dict = {}
for i in point5D:
    seg = i[3]
    bin = i[4]
    if ((seg, bin) in min_z_points_dict):
        if((min_z_points_dict[(seg, bin)])[2] > i[2]):
            min_z_points_dict[(seg, bin)] = i
    else:
        min_z_points_dict[(seg, bin)] = i
min_z_points = []
for i in min_z_points_dict:
    min_z_points.append(min_z_points_dict[i])
min_z_points = np.array(min_z_points)
# min_z_points_sorted = min_z_points[np.argsort(min_z_points[:, 2])]
# min_z_points_sorted = np.array(min_z_points_sorted)


# #------------------------------converting min_z points back into 3D----------------------------------------
# index = [3, 4]
# min_z_points_3D = []
# for i in min_z_points:
#     min_z_points_3D.append(np.delete(i, index))
# min_z_points_3D = np.array(min_z_points_3D)

# #------------Plotting the original point cloud in red and min ground points extracted in yellow in 3D-------------
# a = vel_msg[:,0]
# b = vel_msg[:,1]
# c = vel_msg[:,2]

# x = min_z_points_3D[:,0]
# y = min_z_points_3D[:,1]
# z = min_z_points_3D[:,2]

# plt.style.use('seaborn-poster')
# fig = plt.figure(figsize = (10,10))
# ax = plt.axes(projection='3d')
# ax.grid()
# # ax.scatter(a, b, c, c = 'r', s = 10)
# ax.scatter(x, y, z, c = 'y', s = 10)
# ax.set_title('3D Scatter Plot')

# # Set axes label
# ax.set_xlabel('x', labelpad=2)
# ax.set_ylabel('y', labelpad=2)
# ax.set_zlabel('z', labelpad=2)

# plt.show()

#---------------Converting the 3D minimum z points into 2D (ignoring the y coordinate to apply ransac)-------------
index = [1, 3, 4]
min_z_points_2D = []
for i in min_z_points:
    min_z_points_2D.append(np.delete(i, index))
min_z_points_2D = np.array(min_z_points_2D)


# #--------------------visualising the ground points in 2D after removing y-coordinate----------------------------
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# # Scatter plot
# # plt.scatter(min_z_points_3D[:, 0], min_z_points_3D[:, 2], c='y', cmap='hot')
# plt.scatter(vel_msg[:, 0], vel_msg[:, 2], c='r', cmap='hot')

# # Display the plot
# plt.show()





# -----------------------------------------RANSAC---------------------------------------------
# Create a 2D array of points
points = min_z_points_2D

#number of points
n = min_z_points_2D.shape[0]

# RANSAC parameters
n_iterations = 100  # Number of RANSAC iterations
sample_size = 2  # Number of points to sample in each iteration
inlier_threshold = 0.05  # Maximum distance for a point to be considered an inlier

best_inliers = []
best_model = None

for _ in range(n_iterations):
    # Randomly sample 'sample_size' points
    sample_indices = random.sample(range(n), sample_size)
    sample = points[sample_indices]

    # Fit a model to the sampled points (in this case, a line)
    model = LinearRegression()
    model.fit(sample[:, 0].reshape(-1, 1), sample[:, 1])

    # Compute the distances between all points and the model
    distances = np.abs(model.predict(points[:, 0].reshape(-1, 1)) - points[:, 1])

    # Find inliers (points that are within the threshold)
    inliers = np.where(distances < inlier_threshold)[0]

    # Check if this model is the best one so far
    if len(inliers) > len(best_inliers):
        best_inliers = inliers
        best_model = model

# Extract the best fit line using all the inliers
x_line = np.linspace(-15, 115, 10)
y_line = best_model.predict(x_line.reshape(-1, 1))

# Plot the inliers and outliers
# inliers = points[best_inliers]
# outliers = np.delete(points, best_inliers, axis=0)






#-------------------------------Visualising the model fit of RANSAC------------------------------------
# plt.scatter(inliers[:, 0], inliers[:, 1], color='blue', marker='o', label='Inliers')
# plt.scatter(outliers[:, 0], outliers[:, 1], color='red', marker='x', label='Outliers')
# # Plot the RANSAC fit line
# plt.plot(x_line, y_line, color='green', linewidth=2, label='RANSAC Line')

# # Display the plot
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()





#----------------------find equation of the RANSAC line (a_line)x + (b_line)y + (c_line) = 0---------------------
x1 = x_line[1]
x2 = x_line[2]
y1 = y_line[1]
y2 = y_line[2]
a_line = (y2-y1)
b_line = (x1-x2)
c_line = ((x2-x1)*y1) - ((y2-y1)*x1)

#----------------------getting points from original pointcloud if they don't classify as ground---------------------
removal_threshold = 0.04
vel_msg_postremoval = []
for i in vel_msg:
    dist_from_line = (abs(a_line*i[0] + b_line*i[2] + c_line))/((a_line**2 + b_line**2)**(0.5))
    # print(dist_from_line)
    if(dist_from_line>removal_threshold):
        vel_msg_postremoval.append(i)

vel_msg_postremoval = np.array(vel_msg_postremoval)
print(vel_msg_postremoval)





#------------------------------visualising the non ground points in 3D----------------------------------
# x = vel_msg_postremoval[:,0]
# y = vel_msg_postremoval[:,1]
# z = vel_msg_postremoval[:,2]

# plt.style.use('seaborn-poster')
# fig = plt.figure(figsize = (10,10))
# ax = plt.axes(projection='3d')
# ax.grid()
# ax.scatter(x, y, z, c = 'b', s = 10)
# ax.set_title('3D Scatter Plot')

# # Set axes label
# ax.set_xlabel('x', labelpad=2)
# ax.set_ylabel('y', labelpad=2)
# ax.set_zlabel('z', labelpad=2)

# plt.show()




#-------------------------------forming a pcd from the postremoval array-------------------------
pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(vel_msg_postremoval)
open3d.io.write_point_cloud("/home/harsh/catkin_ws/src/perception_pcl/pcl_ros/tools/non_ground_pcd/c1-376528000_no_ground.pcd", pcd)

