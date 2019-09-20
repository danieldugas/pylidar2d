from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import rospy
import tf
import threading
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, Quaternion
from timeit import default_timer as timer
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist
from std_srvs.srv import Trigger, TriggerResponse

# local packages
from pose2d import Pose2D, apply_tf, apply_tf_to_pose, inverse_pose2d
import lidar_clustering

class Clustering(object):
    def __init__(self, args):
        self.args = args
        # consts
        self.kLidarTopic = "/combined_scan"
        self.kFixedFrameTopic = "/pepper_robot/odom"
        self.kCmdVelTopic = "/cmd_vel"
        self.kGlobalPathTopic = "/planner/GlobalPath"
        self.kFixedFrame = "odom" # ideally something truly fixed, like /map
        self.kRobotFrame = "base_footprint"
        self.kMaxObstacleVel_ms = 10. # [m/s]
        # vars
        self.msg_prev = None
        self.odom = None
        self.tf_rob_in_fix = None
        # ROS
        rospy.init_node('clustering', anonymous=True)
        rospy.Subscriber(self.kLidarTopic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.kFixedFrameTopic, Odometry, self.odom_callback, queue_size=1)
        self.pubs = [rospy.Publisher("debug{}".format(i), LaserScan, queue_size=1) for i in range(3)]
        # tf
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        # Timers
        rospy.Timer(rospy.Duration(0.001), self.tf_callback)
        # visuals 
        self.fig = plt.figure("clusters")
        try:
            plt.show()
#             rospy.spin()
        except KeyboardInterrupt:
            rospy.signal_shutdown('KeyboardInterrupt')


    def odom_callback(self, msg):
        self.odom = msg

    def tf_callback(self, event=None):
        try:
             self.tf_rob_in_fix = self.tf_listener.lookupTransform(self.kFixedFrame, self.kRobotFrame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            return

    def scan_callback(self, msg):
        atic = timer()
        # edge case: first callback
        if self.msg_prev is None:
            self.msg_prev = msg
            return
        if self.odom is None:
            print("odom not received yet")
            return
        if self.tf_rob_in_fix is None:
            print("tf_rob_in_fix not found yet")
            return
        # TODO check that odom and tf are not old

        scan = np.array(msg.ranges, dtype=np.float32)

        # clustering
        EUCLIDEAN_CLUSTERING_THRESH_M = 0.2
        angles = np.linspace(0, 2*np.pi, scan.shape[0]+1, dtype=np.float32)[:-1]
        xx = np.cos(angles) * scan
        yy = np.sin(angles) * scan
        tic = timer()
        clusters, _, _ = lidar_clustering.euclidean_clustering(scan, angles,
                                                         EUCLIDEAN_CLUSTERING_THRESH_M)
        cluster_sizes = lidar_clustering.cluster_sizes(len(scan), clusters)
        toc = timer()
        print("Clustering : {} ms".format((toc-tic)*1000.))

        # center of gravity
        tic = timer()
        cogs = [[np.mean(xx[c]), np.mean(yy[c])] for c in clusters]
        radii = [np.max(np.sqrt((xx[c]-cog[0])**2 + (yy[c]-cog[1])**2)) for cog, c in zip(cogs, clusters)]
        toc = timer()
        print("C.O.Gs, Radii : {} ms".format((toc-tic)*1000.))

        # VISUALS -------------------------------------------

        plt.figure("clusters")
        plt.cla()
        plt.scatter(xx, yy, zorder=1, facecolor=(1,1,1,1), edgecolor=(0,0,0,1), color='k', marker='.')
#         for x, y in zip(xx, yy):
#             plt.plot([0, x], [0, y], linewidth=0.01, color='red' , zorder=2)

        for c in clusters:
            plt.plot(xx[c], yy[c], zorder=2)

        for cog, r in zip(cogs, radii):
            patch = patches.Circle(cog, r, facecolor=(0,0,0,0), edgecolor=(0,0,0,1), linestyle='--')
            plt.gca().add_artist(patch)

        plt.axis('equal')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.pause(0.1)


        if False:
            # publish cmd_vel vis
            pub = rospy.Publisher("/dwa_cmd_vel", Marker, queue_size=1)
            mk = Marker()
            mk.header.frame_id = self.kRobotFrame
            mk.ns = "arrows"
            mk.id = 0
            mk.type = 0
            mk.action = 0
            mk.scale.x = 0.02
            mk.scale.y = 0.02
            mk.color.b = 1
            mk.color.a = 1
            mk.frame_locked = True
            pt = Point()
            pt.x = 0
            pt.y = 0
            pt.z = 0.03
            mk.points.append(pt)
            pt = Point()
            pt.x = 0 + best_u * DWA_DT
            pt.y = 0 + best_v * DWA_DT
            pt.z = 0.03
            mk.points.append(pt)
            pub.publish(mk)
            pub = rospy.Publisher("/dwa_goal", Marker, queue_size=1)
            mk = Marker()
            mk.header.frame_id = self.kRobotFrame
            mk.ns = "arrows"
            mk.id = 0
            mk.type = 0
            mk.action = 0
            mk.scale.x = 0.02
            mk.scale.y = 0.02
            mk.color.g = 1
            mk.color.a = 1
            mk.frame_locked = True
            pt = Point()
            pt.x = 0
            pt.y = 0
            pt.z = 0.03
            mk.points.append(pt)
            pt = Point()
            pt.x = 0 + gx
            pt.y = 0 + gy
            pt.z = 0.03
            mk.points.append(pt)
            pub.publish(mk)
            pub = rospy.Publisher("/dwa_radius", Marker, queue_size=1)
            mk = Marker()
            mk.header.frame_id = self.kRobotFrame
            mk.ns = "radius"
            mk.id = 0
            mk.type = 3
            mk.action = 0
            mk.pose.position.z = -0.1
            mk.scale.x = COMFORT_RADIUS_M * 2
            mk.scale.y = COMFORT_RADIUS_M * 2
            mk.scale.z = 0.01
            mk.color.b = 1
            mk.color.g = 1
            mk.color.r = 1
            mk.color.a = 1
            mk.frame_locked = True
            pub.publish(mk)

            # publish scan prediction
            msg_next = deepcopy(msg)
            msg_next.ranges = s_next
            # for pretty colors
            cluster_ids = clustering.cluster_ids(len(x), clusters)
            random_map = np.arange(len(cluster_ids))
            np.random.shuffle(random_map)
            cluster_ids = random_map[cluster_ids]
            msg_next.intensities = cluster_ids
            self.pubs[0].publish(msg_next)

            # publish past
            msg_prev = deepcopy(msg)
            msg_prev.ranges = self.msg_prev.ranges
            self.pubs[1].publish(msg_prev)

            # ...

        # finally, set up for next callback
        self.msg_prev = msg

        atoc = timer()
        if self.args.hz:
            print("DWA callback: {:.2f} Hz".format(1/(atoc-atic)))


def parse_args():
    import argparse
    ## Arguments
    parser = argparse.ArgumentParser(description='ROS node for clustering 2d lidar')
    parser.add_argument('--hz',
            action='store_true',
            help='if set, prints planner frequency to script output',
            )
    ARGS, unknown_args = parser.parse_known_args()

    # deal with unknown arguments
    # ROS appends some weird args, ignore those, but not the rest
    if unknown_args:
        rosparser = argparse.ArgumentParser()
        rosparser.add_argument(
                '__log:')
        rosparser.add_argument(
                '__name:')
        rosargs = rosparser.parse_args(unknown_args)
    return ARGS

if __name__=="__main__":
    args = parse_args()
    clustering = Clustering(args)
