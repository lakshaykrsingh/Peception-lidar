#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import pcl

def pointcloud_callback(msg):
    pc_data = []
    for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        pc_data.append([point[0], point[1], point[2]])

    if pc_data:
        cloud = pcl.PointCloud()
        cloud.from_list(pc_data)

        # Save the PointCloud data to a PCD file
        pcl.save(cloud, r'/home/harsh/catkin_ws/src/lidar/pcd/raw.pcd')
        rospy.loginfo('Point cloud data saved to raw.pcd')

if __name__ == '__main__':
    rospy.init_node('pointcloud_to_pcd')

    # Subscribe to the PointCloud2 topic you want to convert
    rospy.Subscriber("/velodyne_points", PointCloud2, pointcloud_callback)

    # Spin to keep the node alive
    rospy.spin()
