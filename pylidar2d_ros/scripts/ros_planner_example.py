#!/usr/bin/env python
from copy import deepcopy
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Twist, Quaternion
from geometry_msgs.msg import Point, Point32, Twist, PoseStamped
from map2d_ros_tools import ReferenceMapAndLocalizationManager
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from nav_msgs.msg import Odometry, Path
from pose2d import Pose2D, apply_tf, apply_tf_to_pose, inverse_pose2d, apply_tf_to_vel
import rospy
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger, TriggerResponse
import signal
import sys
import tf
from tf2_ros import TransformException
import threading
from timeit import default_timer as timer
from visualization_msgs.msg import Marker, MarkerArray
from CMap2D import CMap2D

# local packages
import lidar_clustering
from lib_tracking.lidar_tracking import Tracker

N_PAST_POS = 100



class Planning(object):
    def __init__(self, args):
        self.args = args
        # consts
        self.kLidarTopic = args.scan_topic
        self.kRadius = rospy.get_param("~/robot_radius", 0.4)
        self.kIdealVelocity = rospy.get_param("~/ideal_velocity", 0.3)
        self.kFixedFrameTopic = "/pepper_robot/odom"
        self.kCmdVelTopic = "/cmd_vel"
        self.kWaypointTopic = "/global_planner/current_waypoint"
        self.kObstaclesTopic = "/obstacles"
        self.kStaticObstaclesTopic = "/close_nonleg_obstacles"
        self.kFixedFrame = args.fixed_frame # ideally something truly fixed, like /map
        self.kRobotFrame = "base_footprint"
        self.kMaxObstacleVel_ms = 10. # [m/s]
        # vars
        self.odom = None
        self.tf_rob_in_fix = None
        self.tf_wpt_in_fix = None
        self.tracker = Tracker(args)
        self.lock = threading.Lock() # for avoiding race conditions
        self.obstacles_buffer = []
        self.static_obstacles_buffer = None
        self.transport_data = None
        self.STOP = False # DEBUG True
        # ROS
        rospy.init_node('clustering', anonymous=True)
        rospy.Subscriber(self.kObstaclesTopic, ObstacleArrayMsg, self.obstacles_callback, queue_size=1)
        rospy.Subscriber(self.kStaticObstaclesTopic, ObstacleArrayMsg, self.static_obstacles_callback, queue_size=1)
        rospy.Subscriber(self.kFixedFrameTopic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.kWaypointTopic, Marker, self.waypoint_callback, queue_size=1)
        self.pubs = [rospy.Publisher("debug{}".format(i), LaserScan, queue_size=1) for i in range(3)]
        # tf
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        # Localization Manager
        mapname = rospy.get_param("/python_tracker/reference_map_name", "map")
        mapframe = rospy.get_param("/python_tracker/reference_map_frame", "reference_map")
        mapfolder = rospy.get_param("/python_tracker/reference_map_folder", "~/maps")
        self.refmap_manager = ReferenceMapAndLocalizationManager(mapfolder, mapname, mapframe, self.kFixedFrame)
        self.refmap_manager.map_as_closed_obstacles = self.refmap_manager.map_.as_closed_obst_vertices()
        rospy.loginfo("cache computed")
        self.is_tracking_global_path = False
        # Timers
        rospy.Timer(rospy.Duration(0.001), self.tf_callback)
        rospy.Timer(rospy.Duration(0.1), self.planner_routine)
        # visuals 
        try:
#             self.visuals_loop()
            rospy.spin()
        except KeyboardInterrupt:
            print("Keyboard interrupt - shutting down.")
            rospy.signal_shutdown('KeyboardInterrupt')

    def visuals_loop(self):
        while True:
            # PLOT ---------------------------
            if self.transport_data is None:
                continue
            sim, agents, metadata, radii, obstacles, pred_tracks = self.transport_data 
            plt.figure("RVO")
            plt.cla()
            x = [sim.getAgentPosition(agent_no)[0] for agent_no in agents]
            y = [sim.getAgentPosition(agent_no)[1] for agent_no in agents]
            for i in range(len(agents)):
                color = "black"
                color = "green" if metadata[i]['source'] == 'robot' else color
                color = "red" if metadata[i]['source'] == 'dynamic' else color
                color = "blue" if metadata[i]['source'] == 'nonleg' else color
                circle = plt.Circle((x[i], y[i]), radii[i], facecolor=color)
                plt.gcf().gca().add_artist(circle)
                # track
                xx = np.array(pred_tracks[i])[:,0]
                yy = np.array(pred_tracks[i])[:,1]
                plt.plot(xx, yy, color=color)
            for vert in obstacles:
                plt.plot([v[0] for v in vert] + [vert[0][0]], [v[1] for v in vert] + [vert[0][1]])
            plt.axes().set_aspect('equal')
            plt.pause(0.1)
            # -----------------------------

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
    # - slam map or reference map (refmap for now)
    # - -points- clusters of size > N closer than X in scan - which are not dynamic objects (N.I.Y)
    # - tracks not older than Y
    def planner_routine(self, event=None):
        nowstamp = rospy.Time.now()
        if self.tf_wpt_in_fix is None:
            rospy.logwarn_throttle(10., "Global path not available yet")
            return
        if self.tf_rob_in_fix is None:
            rospy.logwarn_throttle(10., "tf_rob_in_fix not available yet")
            return
        if self.odom is None:
            rospy.logwarn_throttle(10., "odom not available yet")
            return
        # remove old obstacles from buffer
        k2s = 2.
        with self.lock:
            tic = timer()
            dynamic_obstacles = []
            self.obstacles_buffer = [obst_msg for obst_msg in self.obstacles_buffer
                    if nowstamp - obst_msg.header.stamp < rospy.Duration(k2s)]
            # DEBUG remove
#             for obst_msg in self.obstacles_buffer:
            for obst_msg in self.obstacles_buffer[-1:]:
                dynamic_obstacles.extend(obst_msg.obstacles)
            # take into account static obstacles if not old
            static_obstacles = []
            if self.static_obstacles_buffer is not None:
                if (nowstamp - self.static_obstacles_buffer.header.stamp) < rospy.Duration(k2s):
                    static_obstacles = self.static_obstacles_buffer.obstacles
            # vel to goal in fix
            goal_in_fix = np.array(Pose2D(self.tf_wpt_in_fix)[:2])
            toc = timer()
            print("Lock preproc {:.1f} ms".format( (toc-tic) * 1000. ))
            # robot position
            tf_rob_in_fix = self.tf_rob_in_fix

        tic = timer()

        # RVO set up
        positions = []
        radii = []
        velocities = []
        obstacles = []
        metadata = [] # agent metadata

        # create agents for RVO
        # first agent is the robot
        positions.append(Pose2D(tf_rob_in_fix)[:2])
        radii.append(self.kRadius)
        velocities.append([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y])
        metadata.append({'source': 'robot'})
        # create agent for every dynamic obstacle
        for obst in dynamic_obstacles:
            positions.append([obst.polygon.points[0].x, obst.polygon.points[0].y])
            radii.append(obst.radius)
            velocities.append( [obst.velocities.twist.linear.x, obst.velocities.twist.linear.y] )
            metadata.append({'source': 'dynamic'})
        # create agents for local static obstacles (table leg, chair, etc)
        for obst in static_obstacles:
            if obst.radius > 0.5:
                continue # TODO big walls make big circles : but dangerous to do this! 
            positions.append([obst.polygon.points[0].x, obst.polygon.points[0].y])
            radii.append(obst.radius)
            velocities.append( [obst.velocities.twist.linear.x, obst.velocities.twist.linear.y] )
            metadata.append({'source': 'nonleg'})

        # create static obstacle vertices from refmap if available (walls, etc)
        if self.refmap_manager.tf_frame_in_refmap is not None:
            kMapObstMaxDist = 3.
            obstacles_in_ref = self.refmap_manager.map_as_closed_obstacles
            pose2d_ref_in_fix = inverse_pose2d(Pose2D(self.refmap_manager.tf_frame_in_refmap))
            near_obstacles = [apply_tf(vertices, pose2d_ref_in_fix) for vertices in obstacles_in_ref 
                              if len(vertices) > 1]
            near_obstacles = [vertices for vertices in near_obstacles 
                              if np.mean(np.linalg.norm(
                                  vertices - np.array(positions[0]), axis=1)) < kMapObstMaxDist]
            obstacles = near_obstacles

        toc = timer()
        print("Pre-RVO {:.1f} ms".format( (toc-tic) * 1000. ))
        tic = timer()

        # RVO ----------------
        import rvo2
        t_horizon = 10.
        DT = 1/10.
        #RVOSimulator(timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity = [0,0]);
        sim = rvo2.PyRVOSimulator(DT, 1.5, 5, 1.5, 2, 0.4, 2)

        # Pass either just the position (the other parameters then use
        # the default values passed to the PyRVOSimulator constructor),
        # or pass all available parameters.
        agents = []
        for p, v, r, m in zip(positions, velocities, radii, metadata):
            #addAgent(posxy, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity = [0,0]);
            a = sim.addAgent(tuple(p), radius=r, velocity=tuple(v))
            agents.append(a)
            # static agents can't move
            if m['source'] != 'robot' and np.linalg.norm(v) == 0:
                sim.setAgentMaxSpeed(a, 0)

        # Obstacle(list of vertices), anticlockwise (clockwise for bounding obstacle)
        for vert in obstacles:
            o1 = sim.addObstacle(list(vert))
        sim.processObstacles()

        toc = timer()
        print("RVO init {:.1f} ms".format( (toc-tic) * 1000. ))
        tic = timer()

        n_steps = int(t_horizon / DT)
        pred_tracks = [[] for a in agents]
        pred_vels = [[] for a in agents]
        pred_t = []
        for step in range(n_steps):
            for i in range(len(agents)):
                # TODO: early stop if agent 0 reaches goal
                a = agents[i]
                pref_vel = velocities[i] # assume agents want to keep initial vel
                if i == 0:
                    vector_to_goal = goal_in_fix - np.array(sim.getAgentPosition(a))
                    pref_vel = self.kIdealVelocity * vector_to_goal / np.linalg.norm(vector_to_goal)
                sim.setAgentPrefVelocity(a, tuple(pref_vel))
                pred_tracks[i].append(sim.getAgentPosition(a))
                pred_vels[i].append(sim.getAgentVelocity(a))
            pred_t.append(1. *step * DT)
            sim.doStep()
        # -------------------------
        toc = timer()
        print("RVO steps {} - {} agents - {} obstacles".format(step, len(agents), len(obstacles)))
        print("RVO sim {:.1f} ms".format( (toc-tic) * 1000. ))


        # PLOT ---------------------------
#         self.transport_data = (sim, agents, metadata, radii, obstacles, pred_tracks)
        # -----------------------------

        # publish predicted tracks as paths
        rob_track = pred_tracks[0]
        pub = rospy.Publisher("/rvo_robot_plan", Path, queue_size=1)
        path_msg = Path()
        path_msg.header.stamp = nowstamp
        path_msg.header.frame_id = self.kFixedFrame
        for pose, t in zip(rob_track, pred_t):
            pose_msg = PoseStamped()
            pose_msg.header.stamp = nowstamp + rospy.Duration(t)
            pose_msg.header.frame_id = path_msg.header.frame_id
            pose_msg.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            path_msg.poses.append(pose_msg)
        pub.publish(path_msg)

        # publish tracks
        pub = rospy.Publisher("/rvo_simulated_tracks", MarkerArray, queue_size=1)
        ma = MarkerArray()
        id_ = 0
        end_x = [sim.getAgentPosition(agent_no)[0] for agent_no in agents]
        end_y = [sim.getAgentPosition(agent_no)[1] for agent_no in agents]
        end_t = pred_t[-1]
        for i in range(len(agents)):
            color = (0,0,0,1) # black
            color = (1,.8,0,1) if metadata[i]['source'] == 'robot' else color # green
            color = (1,0,0,1) if metadata[i]['source'] == 'dynamic' else color # red
            color = (0,0,1,1) if metadata[i]['source'] == 'nonleg' else color # blue
            # track line
            mk = Marker()
            mk.lifetime = rospy.Duration(0.1)
            mk.header.frame_id = self.kFixedFrame
            mk.ns = "tracks"
            mk.id = i
            mk.type = 4 # LINE_STRIP
            mk.action = 0
            mk.scale.x = 0.02
            mk.color.r = color[0]
            mk.color.g = color[1]
            mk.color.b = color[2]
            mk.color.a = color[3]
            mk.frame_locked = True
            xx = np.array(pred_tracks[i])[:,0]
            yy = np.array(pred_tracks[i])[:,1]
            for x, y, t in zip(xx, yy, pred_t):
                pt = Point()
                pt.x = x
                pt.y = y
                pt.z = t / t_horizon
                mk.points.append(pt)
            ma.markers.append(mk)
            # endpoint
            r = radii[i]
            mk = Marker()
            mk.lifetime = rospy.Duration(0.1)
            mk.header.frame_id = self.kFixedFrame
            mk.ns = "endpoints"
            mk.id = i
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
            mk.pose.position.x = end_x[i]
            mk.pose.position.y = end_y[i]
            mk.pose.position.z = end_t / t_horizon
            ma.markers.append(mk)
        pub.publish(ma)


        pred_rob_vel_in_fix = np.array(pred_vels[0][1]) # 2 is a cheat because 0 is usually the current speed
        pred_end_in_fix = np.array([end_x[0], end_y[0]])
        # in robot frame
        pose2d_fix_in_rob = inverse_pose2d(Pose2D(tf_rob_in_fix))
        pred_rob_vel_in_rob = apply_tf_to_vel(np.array(list(pred_rob_vel_in_fix) + [0]),
                                              pose2d_fix_in_rob)[:2]
        pred_end_in_rob = apply_tf(pred_end_in_fix, pose2d_fix_in_rob)
        # resulting command vel, and path
        best_u, best_v = pred_rob_vel_in_rob
        # check if goal is reached
        if np.linalg.norm(pred_end_in_rob) < 0.5:
            best_u, best_v = (0, 0)


        # Slow turn towards goal
        # TODO
        best_w = 0
        WMAX = 0.5
        gx, gy = pred_end_in_rob
        angle_to_goal = np.arctan2(gy, gx) # [-pi, pi]
        if np.sqrt(gx * gx + gy * gy) > 0.5: # turn only if goal is far away
            if np.abs(angle_to_goal) > (np.pi / 4/ 10): # deadzone
                best_w = np.clip(angle_to_goal, -WMAX, WMAX) # linear ramp


        cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        if not self.STOP:
            # publish cmd_vel
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = best_u
            cmd_vel_msg.linear.y = best_v
            cmd_vel_msg.angular.z = best_w
            cmd_vel_pub.publish(cmd_vel_msg)

    def waypoint_callback(self, msg):
        self.tf_timeout = rospy.Duration(0.1)
        """ If a global path is received (in map frame), try to track it """
        with self.lock:
            ref_frame = msg.header.frame_id
            wpt_ref_xy = [msg.pose.position.x, msg.pose.position.y]
            try:
                time = rospy.Time.now()
                tf_info = [self.kFixedFrame, msg.header.frame_id, time]
                self.tf_listener.waitForTransform(*(tf_info + [self.tf_timeout]))
                tf_ref_in_fix = self.tf_listener.lookupTransform(*tf_info)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException,
                    TransformException) as e:
                rospy.logwarn("[{}.{}] tf to refmap frame for time {}.{} not found: {}".format(
                    rospy.Time.now().secs, rospy.Time.now().nsecs, time.secs, time.nsecs, e))
                return
            wpt_fix_xy = apply_tf(np.array(wpt_ref_xy), Pose2D(tf_ref_in_fix))
            self.tf_wpt_in_fix = (np.array([wpt_fix_xy[0], wpt_fix_xy[1], 0.]), # trans
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
