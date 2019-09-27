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

class Planning(object):
    def __init__(self, args):
        self.args = args
        # consts
        self.kLidarTopic = args.scan_topic
        self.kFixedFrameTopic = "/pepper_robot/odom"
        self.kCmdVelTopic = "/cmd_vel"
        self.kGlobalPathTopic = "/planner/GlobalPath"
        self.kObstaclesTopic = "/obstacles"
        self.kStaticObstaclesTopic = "/close_nonleg_obstacles"
        self.kFixedFrame = args.fixed_frame # ideally something truly fixed, like /map
        self.kRobotFrame = "base_footprint"
        self.kMaxObstacleVel_ms = 10. # [m/s]
        # vars
        self.odom = None
        self.tf_rob_in_fix = None
        self.tracker = Tracker(args)
        self.lock = threading.Lock() # for avoiding race conditions
        self.obstacles_buffer = []
        self.static_obstacles_buffer = None
        # ROS
        rospy.init_node('clustering', anonymous=True)
        rospy.Subscriber(self.kLidarTopic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.kObstaclesTopic, ObstacleArrayMsg, self.obstacles_callback, queue_size=1)
        rospy.Subscriber(self.kStaticObstaclesTopic, ObstacleArrayMsg, self.static_obstacles_callback, queue_size=1)
        rospy.Subscriber(self.kFixedFrameTopic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.kGlobalPathTopic, Path, self.global_path_callback, queue_size=1)
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
        rospy.Timer(rospy.Duration(0.1), self.global_path_tracking_routine)
        rospy.Timer(rospy.Duration(0.1), self.planner_routine)
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

    def obstacles_callback(self, msg):
        # Add obstacles to buffer with their associated time
        with self.lock:
            self.obstacles_buffer.append(msg)

    def static_obstacles_callback(self, msg):
        self.static_obstacles_buffer = msg

    # TODO
    # plan based on 
    # - slam map or reference map (slam map for now)
    # - -points- clusters of size > N closer than X in scan - which are not dynamic objects
    # - tracks not older than Y
    def planner_routine(self, event=None):
        # remove old obstacles from buffer
        with self.lock:
            self.obstacles_buffer = [obstacles for obstacles in self.obstacles_buffer
                                if msg.header.stamp - obstacles.header.stamp > rospy.Duration(2.)]
        # create agent for every dynamic obstacle

        # make agent 0 the robot
        t_horizon = 5.
        positions = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        vert = [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]
        vert = np.array(vert) + [1, 0]
        obstacles = [vert]

        # RVO ----------------
        import rvo2
        DT = 1/60.
        #RVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity = [0,0]);
        sim = rvo2.PyRVOSimulator(DT, 1.5, 5, 1.5, 2, 0.4, 2)

        # Pass either just the position (the other parameters then use
        # the default values passed to the PyRVOSimulator constructor),
        # or pass all available parameters.
        agents = []
        for p, v, r in zip(positions, velocities, radii):
            #addAgent(posxy, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity = [0,0]);
            agents.append(sim.addAgent(tuple(p), radius=r, velocity=v))

        # Obstacle(list of vertices), anticlockwise (clockwise for bounding obstacle)
        for vert in obstacles:
            o1 = sim.addObstacle(list(vert))
        sim.processObstacles()


        n_steps = int(t_horizon / DT)
        pred_tracks = [[] for a in agents]
        pred_vels = [[] for a in agents]
        for step in range(n_steps):
            for i in range(len(agents)):
                # TODO: early stop if agent 0 reaches goal
                a = agents[i]
                pref_vel = velocities[i] # assume agents want to keep initial vel
                sim.setAgentPrefVelocity(a, pref_vel)
                pred_tracks[i] = sim.getAgentPosition(a)
                pred_vels[i] = sim.getAgentVelocity(a)
            sim.doStep()
        # -------------------------

        # publish predicted tracks as paths

        # resulting command vel, and path


    def global_path_callback(self, msg):
        """ If a global path is received (in map frame), try to track it """
        with self.lock:
            self.global_path = msg
            self.is_tracking_global_path = True
        # TODO: if new path is longer, update the progress
        # how far is the robot current pos from goal along old path? (can be unclear as we might be off path)
        # how far is the robot current pos from goal along new path? (easy)
        # difference between the two is the path elongation/shortening
        # ...

    def global_path_tracking_routine(self, event):
        """ sets the goal to a waypoint along the global path """
        with self.lock:
            self.WAYPOINT_DIST_STEPS = 5 # steps between closest point on path and waypoint. should translate to ~1-2m real distance. assumes evenly sampled path.
    #         self.WAYPOINT_DIST_M = 1.
            if not self.is_tracking_global_path:
                return
            ## transform path to fixed frame
            try:
                 tf_path_in_fix = self.tf_listener.lookupTransform(
                         self.kFixedFrame,
                         self.global_path.header.frame_id,
                         self.global_path.header.stamp - rospy.Duration(0.1))
                 tf_rob_in_fix = self.tf_listener.lookupTransform(self.kFixedFrame, self.kRobotFrame, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print("Could not set goal from global path:")
                print(e)
                return
            # path in path frame
            gp_xy = np.array(
                    [[pose_st.pose.position.x, pose_st.pose.position.y]
                      for pose_st in self.global_path.poses]
                    )
            # path in fixed frame
            gp_fix_xy = apply_tf(gp_xy, Pose2D(tf_path_in_fix))
            # distance to goal along path
            gp_dtg = np.array([pose_st.pose.position.z for pose_st in self.global_path.poses])
            ## robot xy in fixed frame
            rob_fix_xy =  Pose2D(tf_rob_in_fix)[:2]
            ## check if goal has been reached
            if np.linalg.norm(rob_fix_xy - gp_fix_xy[-1]) < 1.:
                print("global_goal reached")
                # set goal as global goal and stop tracking
                self.is_tracking_global_path = False
                self.waypoint_dtg = None
                wpt_fix_xy = gp_fix_xy[-1]
                self.tf_goal_in_fix = (np.array([wpt_fix_xy[0], wpt_fix_xy[1], 0.]), # trans
                                       tf.transformations.quaternion_from_euler(0,0,0)) # quat
                return
            ## if waypoint is not set yet, set waypoint one meter from the robot along the path
            if self.waypoint_dtg is None:
                # walk N steps forwards from start of path
                waypoint_id = min(len(gp_dtg)-1, self.WAYPOINT_DIST_STEPS)
                self.waypoint_dtg = gp_dtg[waypoint_id]
                # set waypoint 
                wpt_fix_xy = gp_fix_xy[waypoint_id]
                self.tf_goal_in_fix = (np.array([wpt_fix_xy[0], wpt_fix_xy[1], 0.]), # trans
                                       tf.transformations.quaternion_from_euler(0,0,0)) # quat
                return
            ## check if we have progressed
            # find closest point to robot along the path
            distances = np.linalg.norm(rob_fix_xy - gp_fix_xy, axis=-1)
            closest_dist = np.min(distances)
            # adds slack in case path has weird shape
            closest_points_ids = np.where(np.abs(distances - closest_dist) < 0.5)[0]
            closest_point_id = np.max(closest_points_ids) # pick most optimistic closest point
            closest_dist = distances[closest_point_id]
            # ass. path is evenly sampled, walk N points towards goal to find next potential waypoint
            tentative_wpt_id = min(len(gp_fix_xy)-1, closest_point_id + self.WAYPOINT_DIST_STEPS)
            tentative_wpt_dtg = gp_dtg[tentative_wpt_id]
            #######
            pub = rospy.Publisher("/tentative_wpt", Marker, queue_size=1)
            mk = Marker()
            mk.header.frame_id = self.kFixedFrame
            mk.ns = "wpt"
            mk.id = 0
            mk.type = 1
            mk.action = 0
            wpt_fix_xy = gp_fix_xy[tentative_wpt_id]
            mk.pose.position.x = wpt_fix_xy[0]
            mk.pose.position.y = wpt_fix_xy[1]
            mk.pose.position.z = -0.09
            mk.scale.x = 0.1
            mk.scale.y = 0.1 
            mk.scale.z = 0.01
            mk.color.b = 0
            mk.color.g = 1
            mk.color.r = 1
            mk.color.a = 1
            mk.frame_locked = True
            pub.publish(mk)
            #########3
            # set as new waypoint
            self.waypoint_dtg = tentative_wpt_dtg
            wpt_fix_xy = gp_fix_xy[tentative_wpt_id]
            self.tf_goal_in_fix = (np.array([wpt_fix_xy[0], wpt_fix_xy[1], 0.]), # trans
                                   tf.transformations.quaternion_from_euler(0,0,0)) # quat

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
    clustering = Planning(args)
