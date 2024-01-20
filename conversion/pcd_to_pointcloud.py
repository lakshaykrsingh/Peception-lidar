#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import pcl

def publish_pointcloud_from_pcd():
    rospy.init_node('pcd_to_pointcloud')
    pub = rospy.Publisher('/cloud_pcd', PointCloud2, queue_size=10)

    # Load the PCD file
    pcd = pcl.load(r"/home/harsh/catkin_ws/src/lidar/pcd/cloud_cluster_0005.pcd")

    # Create a PointCloud2 message
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "base_link"  # Set the appropriate frame ID
    pc_data = [(point[0], point[1], point[2]) for point in pcd]
    pc2_msg = point_cloud2.create_cloud_xyz32(header, pc_data)

    rate = rospy.Rate(10)  # Adjust the publishing rate as needed

    while not rospy.is_shutdown():
        pub.publish(pc2_msg)
        rospy.loginfo('Publishing pcd to /cloud_pcd topic')
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_pointcloud_from_pcd()
    except rospy.ROSInterruptException:
        pass
