#!/usr/bin/env python3
import rospy
import open3d as o3d
import numpy as np

def main():
    # Read in the cloud data
    cloud = o3d.io.read_point_cloud("/home/harsh/catkin_ws/src/lidar/pcd/raw.pcd")
    # print(f"PointCloud before filtering has: {len(np.asarray(cloud.points))} data points.")

    # Create the segmentation object for the planar model and set all the parameters
    plane_model, inliers = cloud.segment_plane(distance_threshold=0.02, ransac_n=100, num_iterations=1000)

    if len(inliers) == 0:
        rospy.loginfo('Could not estimate a planar model for the given dataset.')
        return

    # Extract the planar inliers from the input cloud
    cloud_plane = cloud.select_by_index(inliers)
    # print(f"PointCloud representing the planar component: {len(np.asarray(cloud_plane.points))} data points.")

    # Remove the planar inliers, extract the rest
    cloud = cloud.select_by_index(inliers, invert=True)

    o3d.io.write_point_cloud("/home/harsh/catkin_ws/src/lidar/pcd/post_ground_removal_3d.pcd", cloud)

if __name__ == "__main__":
    rospy.init_node("ransac3d_ground_removal")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        try:
            main()
            rospy.loginfo('Performing 3D RANSAC ground_removal')
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
