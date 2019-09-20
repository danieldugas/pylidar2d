from copy import deepcopy
from matplotlib import pyplot as plt
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
import clustering

class Responsive(object):
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
        self.tf_goal_in_fix = None
        self.lock = threading.Lock() # for avoiding race conditions
        self.STOP = True # disables autonomous control
        if args.no_stop:
            self.STOP = False
        self.is_tracking_global_path = False
        self.waypoint_dtg = None # DTG of current waypoint along global path. can only decrease if path is constant.
        # ROS
        rospy.init_node('responsive', anonymous=True)
        rospy.Subscriber(self.kGlobalPathTopic, Path, self.global_path_callback, queue_size=1)
        rospy.Subscriber(self.kLidarTopic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.kFixedFrameTopic, Odometry, self.odom_callback, queue_size=1)
        self.pubs = [rospy.Publisher("debug{}".format(i), LaserScan, queue_size=1) for i in range(3)]
        self.cmd_vel_pub = rospy.Publisher(self.kCmdVelTopic, Twist, queue_size=1)
        # tf
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        # Timers
        rospy.Timer(rospy.Duration(0.001), self.tf_callback)
        rospy.Timer(rospy.Duration(0.1), self.global_path_tracking_routine)
        # Services
        rospy.Service('stop_autonomous_motion', Trigger, 
                self.stop_autonomous_motion_service_call)
        rospy.Service('resume_autonomous_motion', Trigger, 
                self.resume_autonomous_motion_service_call)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            # publish cmd_vel
            cmd_vel_msg = Twist()
            self.cmd_vel_pub.publish(cmd_vel_msg)
            rospy.signal_shutdown('KeyboardInterrupt')

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
            # if waypoint is closer to goal, set as new waypoint
            if tentative_wpt_dtg < self.waypoint_dtg:
                self.waypoint_dtg = tentative_wpt_dtg
                wpt_fix_xy = gp_fix_xy[tentative_wpt_id]
                self.tf_goal_in_fix = (np.array([wpt_fix_xy[0], wpt_fix_xy[1], 0.]), # trans
                                       tf.transformations.quaternion_from_euler(0,0,0)) # quat
                return
            # detect potential path failures
            if (tentative_wpt_dtg - self.waypoint_dtg) > 5.:
                # we seem to have moved very far backwards along the path. What happened?
                print("Robot has veered far away ({} - {} = {} > 5) from path. Possible path switch occured?".format(
                   tentative_wpt_dtg, self.waypoint_dtg,  tentative_wpt_dtg - self.waypoint_dtg))
            if np.linalg.norm(rob_fix_xy - self.tf_goal_in_fix[0][:2]) < 0.5:
                # it seems our current waypoint is no longer valid. Reset dtg
                self.waypoint_dtg = None


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
        if self.tf_goal_in_fix is None:
            self.tf_goal_in_fix = self.tf_rob_in_fix
            print("goal set")
        # TODO check that odom and tf are not old

        # measure rotation TODO
        s = np.array(msg.ranges)

        # prediction
        dt = (msg.header.stamp - self.msg_prev.header.stamp).to_sec()
        s_prev =  np.array(self.msg_prev.ranges)
        ds = (s - s_prev)
        max_ds = self.kMaxObstacleVel_ms * dt
        ds_capped = ds
        ds_capped[np.abs(ds) > max_ds] = 0
        s_next = np.maximum(0, s + ds_capped).astype(np.float32)

        # cluster
        EUCLIDEAN_CLUSTERING_THRESH_M = 0.05
        angles = np.linspace(0, 2*np.pi, s_next.shape[0]+1, dtype=np.float32)[:-1]
        clusters, x, y = clustering.euclidean_clustering(s_next, angles,
                                                         EUCLIDEAN_CLUSTERING_THRESH_M)
        cluster_sizes = clustering.cluster_sizes(len(s_next), clusters)
        s_next[cluster_sizes <= 3] = 25


        # dwa
        # Get state
        # goal in robot frame
        goal_in_robot_frame = apply_tf_to_pose(Pose2D(self.tf_goal_in_fix), inverse_pose2d(Pose2D(self.tf_rob_in_fix)))
        gx = goal_in_robot_frame[0]
        gy = goal_in_robot_frame[1]
        
        # robot speed in robot frame
        u = self.odom.twist.twist.linear.x
        v = self.odom.twist.twist.linear.y
        w = self.odom.twist.twist.angular.z


        DWA_DT = 0.5
        COMFORT_RADIUS_M = 0.7
        tic = timer()
        best_u, best_v, best_score = dynamic_window.linear_dwa(s_next,
            angles,
            u, v, gx, gy, DWA_DT,
            DV=0.05,
            UMIN=-0.5,
            UMAX=0.5,
            VMIN=-0.5,
            VMAX=0.5,
            AMAX=10.,
            COMFORT_RADIUS_M=COMFORT_RADIUS_M,
            )
        toc = timer()
#         print(best_u * DWA_DT, best_v * DWA_DT, best_score)
#         print("DWA: {:.2f} Hz".format(1/(toc-tic)))

        # Slow turn towards goal
        # TODO
        best_w = 0
        WMAX = 0.5
        angle_to_goal = np.arctan2(gy, gx) # [-pi, pi]
        if np.sqrt(gx * gx + gy * gy) > 0.5: # turn only if goal is far away
            if np.abs(angle_to_goal) > (np.pi / 4/ 10): # deadzone
                best_w = np.clip(angle_to_goal, -WMAX, WMAX) # linear ramp


        if not self.STOP:
            # publish cmd_vel
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = best_u * 0.5
            cmd_vel_msg.linear.y = best_v * 0.5
            cmd_vel_msg.angular.z = best_w
            self.cmd_vel_pub.publish(cmd_vel_msg)

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


    def stop_autonomous_motion_service_call(self, req):
        with self.lock:
            if not self.STOP:
                print("Surrendering robot control")
            self.STOP = True
        return TriggerResponse(True, "")

    def resume_autonomous_motion_service_call(self, req):
        with self.lock:
            if self.STOP:
                print("Assuming robot control")
            self.STOP = False
        return TriggerResponse(True, "")

def parse_args():
    import argparse
    ## Arguments
    parser = argparse.ArgumentParser(description='Responsive motion planner for pepper')
    parser.add_argument('--no-stop',
            action='store_true',
            help='if set, the planner will immediately send cmd_vel instead of waiting for hand-over',
            )
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
    responsive = Responsive(args)
