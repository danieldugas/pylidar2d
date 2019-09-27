#!/usr/bin/env python
from copy import deepcopy
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Twist, Quaternion
from geometry_msgs.msg import Point, Point32, Twist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from nav_msgs.msg import Odometry, Path
from pose2d import Pose2D, apply_tf, apply_tf_to_pose, inverse_pose2d
import rospy
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger, TriggerResponse
import signal
import sys
import tf
import threading
from timeit import default_timer as timer
from visualization_msgs.msg import Marker, MarkerArray
from CMap2D import CMap2D

# local packages
import lidar_clustering
from lib_tracking.lidar_tracking import Tracker

N_PAST_POS = 100


class ReferenceMapAndLocalizationManager(object):
    """ If a reference map is provided and a tf exists,
    keeps track of the tf for given frame in the reference map frame """
    def __init__(self, map_folder, map_filename, reference_map_frame, frame_to_track):
        self.tf_frame_in_refmap = None
        self.map_ = None
        self.map_as_tsdf = None
        # loads map based on ros params
        folder = map_folder
        filename =  map_filename
        try:
            self.map_ = CMap2D(folder, filename)
        except IOError as e:
            rospy.logwarn(rospy.get_namespace())
            rospy.logwarn("Failed to load reference map. Make sure {}.yaml and {}.pgm"
                   " are in the {} folder.".format(filename, filename, folder))
            rospy.logwarn("Disabling. No global localization or reference map will be available.")
            return
        # get frame info
        self.kRefMapFrame = reference_map_frame
        self.kFrame = frame_to_track
        # launch callbacks
        self.tf_listener = tf.TransformListener()
        rospy.Timer(rospy.Duration(0.01), self.tf_callback)
        self.map_as_tsdf = self.map_.as_sdf()

    def tf_callback(self, event=None):
        try:
             self.tf_frame_in_refmap = self.tf_listener.lookupTransform(self.kRefMapFrame, self.kFrame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(10., e)
            return

class Clustering(object):
    def __init__(self, args):
        self.args = args
        # consts
        self.kLidarTopic = args.scan_topic
        self.kFixedFrameTopic = "/pepper_robot/odom"
        self.kCmdVelTopic = "/cmd_vel"
        self.kGlobalPathTopic = "/planner/GlobalPath"
        self.kFixedFrame = args.fixed_frame # ideally something truly fixed, like /map
        self.kRobotFrame = "base_footprint"
        self.kMaxObstacleVel_ms = 10. # [m/s]
        # vars
        self.odom = None
        self.tf_rob_in_fix = None
        self.tracker = Tracker(args)
        self.lock = threading.Lock() # for avoiding race conditions
        # ROS
        rospy.init_node('clustering', anonymous=True)
        rospy.Subscriber(self.kLidarTopic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.kFixedFrameTopic, Odometry, self.odom_callback, queue_size=1)
        self.pubs = [rospy.Publisher("debug{}".format(i), LaserScan, queue_size=1) for i in range(3)]
        # tf
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        # Localization Manager
        mapname = rospy.get_param("/python_tracker/reference_map_name", "map")
        mapframe = rospy.get_param("/python_tracker/reference_map_frame", "reference_map")
        mapfolder = rospy.get_param("/python_tracker/reference_map_folder", "~/maps")
        self.refmap_manager = ReferenceMapAndLocalizationManager(mapfolder, mapname, mapframe, self.kFixedFrame)
        # Timers
        rospy.Timer(rospy.Duration(0.001), self.tf_callback)
        # data
        self.transport_data = None
        self.tf_pastrobs_in_fix = []
        # visuals 
        self.fig = plt.figure("clusters")
        try:
#             self.visuals_loop()
            rospy.spin()
        except KeyboardInterrupt:
            print("Keyboard interrupt - shutting down.")
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
        with self.lock:
            # edge case: first callback
            if self.odom is None:
                print("odom not received yet")
                return
            if self.tf_rob_in_fix is None:
                print("tf_rob_in_fix not found yet")
                return
            # TODO check that odom and tf are not old

            self.tf_pastrobs_in_fix.append(self.tf_rob_in_fix)
            if len(self.tf_pastrobs_in_fix) > 2*N_PAST_POS:
                self.tf_pastrobs_in_fix = self.tf_pastrobs_in_fix[::2]

            scan = np.array(msg.ranges, dtype=np.float32)

            # clustering
            EUCLIDEAN_CLUSTERING_THRESH_M = 0.05
            MIN_CLUSTER_SIZE = 3
            angles = np.linspace(0, 2*np.pi, scan.shape[0]+1, dtype=np.float32)[:-1]
            xx = np.cos(angles) * scan
            yy = np.sin(angles) * scan
            tic = timer()
            clusters, _, _ = lidar_clustering.euclidean_clustering(scan, angles,
                                                             EUCLIDEAN_CLUSTERING_THRESH_M)
            cluster_sizes = lidar_clustering.cluster_sizes(len(scan), clusters)
            toc = timer()
            clustering_time = toc-tic

            # filter small clusters
    #         clusters = [c for c, l in zip(clusters, cluster_sizes) if l >= MIN_CLUSTER_SIZE]

            # center of gravity
            tic = timer()
            cogs = [np.array([np.mean(xx[c]), np.mean(yy[c])]) for c in clusters]
            radii = [np.max(np.sqrt((xx[c]-cog[0])**2 + (yy[c]-cog[1])**2)) for cog, c in zip(cogs, clusters)]
            toc = timer()
            cog_radii_time = toc-tic

            # in fixed frame
            cogs_in_fix = []
            for i in range(len(cogs)):
                cogs_in_fix.append(apply_tf(cogs[i], Pose2D(self.tf_rob_in_fix)))

            # legs
            is_legs = []
            for r, c, cog_in_fix in zip(radii, clusters, cogs_in_fix):
                is_leg = True if 0.03 < r and r < 0.15 and len(c) >= MIN_CLUSTER_SIZE else False
                # if a reference map and localization are available, filter detections which are in map
                if self.refmap_manager.tf_frame_in_refmap is not None:
                    cog_in_refmap = apply_tf(cog_in_fix, Pose2D(self.refmap_manager.tf_frame_in_refmap))
                    ij = cog_in_refmap_ij = self.refmap_manager.map_.xy_to_ij([cog_in_refmap])[0]
                    # the distance between the c.o.g. and the nearest obstacle in the refmap
                    if self.refmap_manager.map_as_tsdf[ij[0], ij[1]] < r:
                        is_leg = False
                is_legs.append(is_leg)


            timestamp = msg.header.stamp
            self.transport_data = (timestamp, xx, yy, clusters, is_legs, cogs, radii, self.tf_rob_in_fix)

            leg_cogs_in_fix = []
            leg_radii = []
            for i in range(len(is_legs)):
                if is_legs[i]:
                    leg_cogs_in_fix.append(cogs_in_fix[i])
                    leg_radii.append(radii[i])

            # Tracks
            self.tracker.match(leg_cogs_in_fix, leg_radii, msg.header.stamp)

        # publish markers
        self.publish_markers()
        self.publish_obstacles()

        atoc = timer()
        if self.args.hz:
            print("DWA callback: {:.2f} Hz | C.O.Gs, Radii: {:.1f} ms, Clustering: {:.1f} ms".format(
                1/(atoc-atic), cog_radii_time*1000., clustering_time*1000.))

    def visuals_loop(self):
        while True:
            with self.lock:
                if self.transport_data is None:
                    plt.pause(0.1)
                    rospy.timer.sleep(0.01)
                    continue
                _, xx, yy, clusters, is_legs, cogs, radii, tf_rob_in_fix = self.transport_data
                # transform data to fixed frame
                pose2d_rob_in_fix = Pose2D(tf_rob_in_fix)
                xy_f = apply_tf(np.vstack([xx, yy]).T, pose2d_rob_in_fix)
                cogs_f = apply_tf(np.array(cogs), pose2d_rob_in_fix)
                # past positions
                pose2d_past_in_fix = []
                if self.tf_pastrobs_in_fix:
                    for tf_a in self.tf_pastrobs_in_fix[::-1]:
                        pose2d_past_in_fix.append(Pose2D(tf_a))
            # Plotting
            plt.figure("clusters")
            plt.cla()
            #   Robot
            plt.scatter([pose2d_rob_in_fix[0]],[pose2d_rob_in_fix[1]],marker='+', color='k',s=1000)
            #   Lidar
#             plt.scatter(xy_f[:,0], xy_f[:,1], zorder=1, facecolor=(1,1,1,1), edgecolor=(0,0,0,1), color='k', marker='.')
#             for x, y in zip(xx, yy):
#                 plt.plot([0, x], [0, y], linewidth=0.01, color='red' , zorder=2)
            #   clusters
#             for c in clusters:
#                 plt.plot(xx[c], yy[c], zorder=2)
            for l, cog, r in zip(is_legs,cogs_f, radii):
                if l:
#                     patch = patches.Circle(cog, r, facecolor=(0,0,0,0), edgecolor=(0,0,0,1), linewidth=3)
#                     plt.gca().add_artist(patch)
                    plt.scatter([cog[0]], [cog[1]], marker='+', color='green', s=500)
                else:
                    patch = patches.Circle(cog, r, facecolor=(0,0,0,0), edgecolor=(0,0,0,1), linestyle='--')
                    plt.gca().add_artist(patch)
            if pose2d_past_in_fix:
                xy_past = np.array(pose2d_past_in_fix)
                plt.plot(xy_past[:,0], xy_past[:,1], color='red')
            # Tracks
            with self.lock:
                self.tracker.vizualize_tracks()
            # Display options
            plt.gca().set_aspect('equal',adjustable='box')
#             LIM = 10
#             plt.ylim([-LIM, LIM])
#             plt.xlim([-LIM, LIM])
            # pause
            plt.pause(0.1)
            rospy.timer.sleep(0.01)

    def publish_markers(self):
        with self.lock:
            if self.transport_data is None:
                return
            timestamp, xx, yy, clusters, is_legs, cogs, radii, tf_rob_in_fix = self.transport_data
            # transform data to fixed frame
            pose2d_rob_in_fix = Pose2D(tf_rob_in_fix)
            xy_f = apply_tf(np.vstack([xx, yy]).T, pose2d_rob_in_fix)
            cogs_f = apply_tf(np.array(cogs), pose2d_rob_in_fix)
            # past positions
            pose2d_past_in_fix = []
            if self.tf_pastrobs_in_fix:
                for tf_a in self.tf_pastrobs_in_fix[::-1]:
                    pose2d_past_in_fix.append(Pose2D(tf_a))
            # tracks
            tracks_xy, tracks_color, tracks_in_frame = [], [], []
            tracks_avg_r = []
            for trackid in self.tracker.active_tracks:
                track = self.tracker.active_tracks[trackid]
                xy = np.array(track.pos_history)
                is_track_in_frame = True
                if trackid in self.tracker.latest_matches:
                    color = (0.,1.,0.,1.) # green
                elif trackid in self.tracker.new_tracks:
                    color = (0.,0.,1.,1.) # blue
                else:
                    color = (0.7, 0.7, 0.7, 1.)
                    is_track_in_frame = False
                tracks_xy.append(xy)
                tracks_color.append(color)
                tracks_in_frame.append(is_track_in_frame)
                tracks_avg_r.append(track.avg_radius())

        pub = rospy.Publisher("/detections", Marker, queue_size=1)
        # ...

        pub = rospy.Publisher("/tracks", MarkerArray, queue_size=1)
        # delete all markers
        ma = MarkerArray()
        mk = Marker()
        mk.header.frame_id = self.kFixedFrame
        mk.ns = "tracks"
        mk.id = 0
        mk.type = 4 # LINE_STRIP
        mk.action = 3 # deleteall
        ma.markers.append(mk)
        pub.publish(ma)
        ma = MarkerArray()
        mk = Marker()
        mk.header.frame_id = self.kFixedFrame
        mk.ns = "tracks"
        mk.id = 0
        mk.type = 3 # CYLINDER
        mk.action = 3 # deleteall
        ma.markers.append(mk)
        pub.publish(ma)
        # publish tracks
        ma = MarkerArray()
        id_ = 0
        # track lines
        for color, xy in zip(tracks_color, tracks_xy):
            mk = Marker()
            mk.header.frame_id = self.kFixedFrame
            mk.ns = "tracks"
            mk.id = id_
            id_+=1
            mk.type = 4 # LINE_STRIP
            mk.action = 0
            mk.scale.x = 0.02
            mk.color.r = color[0]
            mk.color.g = color[1]
            mk.color.b = color[2]
            mk.color.a = color[3]
            mk.frame_locked = True
            for x, y in xy:
                pt = Point()
                pt.x = x
                pt.y = y
                pt.z = 0.03
                mk.points.append(pt)
            ma.markers.append(mk)
        # track endpoint
        for color, xy, r in zip(tracks_color, tracks_xy, tracks_avg_r):
            x, y = xy[-1]
            mk = Marker()
            mk.header.frame_id = self.kFixedFrame
            mk.ns = "tracks"
            mk.id = id_
            id_+=1
            mk.type = 3 # CYLINDER
            mk.action = 0
            mk.scale.x = r * 2.
            mk.scale.y = r * 2.
            mk.scale.z = 0.1
            mk.color.r = color[0]
            mk.color.g = color[1]
            mk.color.b = color[2]
            mk.color.a = color[3]
            mk.frame_locked = True
            mk.pose.position.x = x
            mk.pose.position.y = y
            mk.pose.position.z = 0.03
            ma.markers.append(mk)
        pub.publish(ma)

        pub = rospy.Publisher("/robot_track", Marker, queue_size=1)
        mk = Marker()
        mk.header.frame_id = self.kFixedFrame
        mk.ns = "robot_track"
        mk.id = 0
        mk.type = 4 # LINE_STRIP
        mk.action = 0
        mk.scale.x = 0.02
        mk.color.r = 1
        mk.color.g = 0
        mk.color.b = 0
        mk.color.a = 1
        mk.frame_locked = True
        for x, y, _ in pose2d_past_in_fix:
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = 0.03
            mk.points.append(pt)
        ma.markers.append(mk)
        pub.publish(mk)

    def publish_obstacles(self):
        with self.lock:
            if self.transport_data is None:
                return
            # non-leg obstacles
            timestamp, xx, yy, clusters, is_legs, cogs, radii, tf_rob_in_fix = self.transport_data
            # tracks
            track_ids = []
            tracks_latest_pos, tracks_color = [], []
            tracks_in_frame, tracks_velocities = [], []
            tracks_radii = []
            for trackid in self.tracker.active_tracks:
                track = self.tracker.active_tracks[trackid]
                xy = np.array(track.pos_history[-1])
                is_track_in_frame = True
                if trackid in self.tracker.latest_matches:
                    color = (0.,1.,0.,1.) # green
                elif trackid in self.tracker.new_tracks:
                    color = (0.,0.,1.,1.) # blue
                else:
                    color = (0.7, 0.7, 0.7, 1.)
                    is_track_in_frame = False
                track_ids.append(trackid)
                tracks_latest_pos.append(xy)
                tracks_color.append(color)
                tracks_in_frame.append(is_track_in_frame)
                tracks_velocities.append(track.estimate_velocity())
                tracks_radii.append(track.avg_radius())

        pub = rospy.Publisher('/obstacles', ObstacleArrayMsg, queue_size=1)
        obstacles_msg = ObstacleArrayMsg() 
        obstacles_msg.header.stamp =  timestamp
        obstacles_msg.header.frame_id = self.kFixedFrame
        for trackid, xy, in_frame, vel, radius in zip(track_ids, tracks_latest_pos, tracks_in_frame, tracks_velocities, tracks_radii):
            if not in_frame:
                continue
            # Add point obstacle
            obst = ObstacleMsg()
            obst.id = trackid
            obst.polygon.points = [Point32()]
            obst.polygon.points[0].x = xy[0]
            obst.polygon.points[0].y = xy[1]
            obst.polygon.points[0].z = 0

            obst.radius = radius

            yaw = np.arctan2(vel[1], vel[0])
            q = tf.transformations.quaternion_from_euler(0,0,yaw)
            obst.orientation = Quaternion(*q)

            obst.velocities.twist.linear.x = vel[0]
            obst.velocities.twist.linear.y = vel[1]
            obst.velocities.twist.linear.z = 0
            obst.velocities.twist.angular.x = 0
            obst.velocities.twist.angular.y = 0
            obst.velocities.twist.angular.z = 0
            obstacles_msg.obstacles.append(obst)
        pub.publish(obstacles_msg)

        pub = rospy.Publisher('/close_nonleg_obstacles', ObstacleArrayMsg, queue_size=1)
        MAX_DIST_STATIC_CLUSTERS_M = 3.
        cogs_in_fix = []
        for i in range(len(cogs)):
            cogs_in_fix.append(apply_tf(cogs[i], Pose2D(tf_rob_in_fix)))
        obstacles_msg = ObstacleArrayMsg() 
        obstacles_msg.header.stamp = timestamp
        obstacles_msg.header.frame_id = self.kFixedFrame
        for cog_in_fix, cog_in_rob, r, is_leg in zip(cogs_in_fix, cogs, radii, is_legs):
            # leg obstacles are already published in the tracked obstacles topic
            if is_leg:
                continue
            # close obstacles only
            if np.linalg.norm(cog_in_rob) > MAX_DIST_STATIC_CLUSTERS_M:
                continue
            # Add point obstacle
            obst = ObstacleMsg()
            obst.id = trackid
            obst.polygon.points = [Point32()]
            obst.polygon.points[0].x = cog_in_fix[0]
            obst.polygon.points[0].y = cog_in_fix[1]
            obst.polygon.points[0].z = 0

            obst.radius = r

            yaw = 0
            q = tf.transformations.quaternion_from_euler(0,0,yaw)
            obst.orientation = Quaternion(*q)

            obst.velocities.twist.linear.x = 0
            obst.velocities.twist.linear.y = 0
            obst.velocities.twist.linear.z = 0
            obst.velocities.twist.angular.x = 0
            obst.velocities.twist.angular.y = 0
            obst.velocities.twist.angular.z = 0
            obstacles_msg.obstacles.append(obst)
        pub.publish(obstacles_msg)

        pub = rospy.Publisher('/obstacle_markers', MarkerArray, queue_size=1)
        # delete all markers
        ma = MarkerArray()
        mk = Marker()
        mk.header.frame_id = self.kFixedFrame
        mk.ns = "obstacles"
        mk.id = 0
        mk.type = 0 # ARROW
        mk.action = 3 # deleteall
        ma.markers.append(mk)
        pub.publish(ma)
        # publish tracks
        ma = MarkerArray()
        id_ = 0
        # track endpoint
        for trackid, xy, in_frame, vel in zip(track_ids, tracks_latest_pos, tracks_in_frame, tracks_velocities):
            if not in_frame:
                continue
            normvel = np.linalg.norm(vel)
            if normvel == 0:
                continue
            mk = Marker()
            mk.header.frame_id = self.kFixedFrame
            mk.ns = "tracks"
            mk.id = trackid
            mk.type = 0 # ARROW
            mk.action = 0
            mk.scale.x = np.linalg.norm(vel)
            mk.scale.y = 0.1
            mk.scale.z = 0.1
            mk.color.r = color[0]
            mk.color.g = color[1]
            mk.color.b = color[2]
            mk.color.a = color[3]
            mk.frame_locked = True
            mk.pose.position.x = xy[0]
            mk.pose.position.y = xy[1]
            mk.pose.position.z = 0.03
            yaw = np.arctan2(vel[1], vel[0])
            q = tf.transformations.quaternion_from_euler(0,0,yaw)
            mk.pose.orientation = Quaternion(*q)
            ma.markers.append(mk)
        pub.publish(ma)


def parse_args():
    import argparse
    ## Arguments
    parser = argparse.ArgumentParser(description='ROS node for clustering 2d lidar')
    parser.add_argument('--hz',
            action='store_true',
            help='if set, prints planner frequency to script output',
            )
    parser.add_argument('--fixed-frame', default="odom")
    parser.add_argument('--scan-topic', default="/combined_scan")
    ARGS, unknown_args = parser.parse_known_args()

    # deal with unknown arguments
    # ROS appends some weird args, ignore those, but not the rest
    if unknown_args:
        non_ros_unknown_args = rospy.myargv(unknown_args)
        if non_ros_unknown_args:
            print("unknown arguments:")
            print(non_ros_unknown_args)
            parser.parse_args(args=["--help"])
            raise ValueError
    return ARGS

if __name__=="__main__":
    args = parse_args()
    clustering = Clustering(args)
