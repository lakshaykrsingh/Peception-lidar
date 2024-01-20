#!/usr/bin/env python3
import rospy
import open3d as o3d
import numpy as np
import os
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import pcl


def main():

    #creating a list having filenames of all the clusters
    rospy.init_node('cone_pose')
    prefixed = [filename for filename in os.listdir(r"/home/harsh/catkin_ws/src/lidar/clusters") if filename.startswith("cloud_cluster")]
    
    #final positions (1 per cone) will be stored in this list
    final_points = []


    #looping over all the clusters. for each, taking average of all points, and storing the avg
    for j in prefixed:
        path = '/home/harsh/catkin_ws/src/lidar/clusters/' + j

        # Read in the cloud data
        cloud = o3d.io.read_point_cloud(path)
        points_arr = np.array(cloud.points)
    
        x_avg=0
        for i in points_arr[:, 0]:
            x_avg+=i
        x_avg = x_avg/len(points_arr)

        y_avg=0
        for i in points_arr[:, 1]:
            y_avg+=i
        y_avg = y_avg/len(points_arr)

        z_avg=0
        for i in points_arr[:, 2]:
            z_avg+=i
        z_avg = z_avg/len(points_arr)

        final_points.append([x_avg, y_avg, z_avg])
    

    #creating a PointCloud2 msg from the list and publishing it to /cone_pose if you want to visualise in rviz
    cloud_ros = pcl.PointCloud()
    cloud_ros.from_list(final_points)
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "base_link"  # Set the appropriate frame ID
    pc_data = [(point[0], point[1], point[2]) for point in cloud_ros]
    pc2_msg = point_cloud2.create_cloud_xyz32(header, pc_data)


    pub = rospy.Publisher('/cone_pose', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)  # Adjust the publishing rate as needed

    while not rospy.is_shutdown():
        pub.publish(pc2_msg)
        rospy.loginfo('Publishing to /cone_pose topic')
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass