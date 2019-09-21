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

N_PAST_POS = 100

class IdCounter(object):
    def __init__(self):
        self.id_count = 0
    def next_id(self):
        id_ = self.id_count
        self.id_count += 1
        return id_

class Track(object):
    def __init__(self, id_):
        self.id = id_
        self.n_hits = 0
        self.sum_radius = 0
        self.pos_history = []
        self.time_history = []

    def add_detection(self, pos, radius, time):
        self.n_hits += 1
        self.sum_radius += radius
        self.pos_history.append(pos)
        self.time_history.append(time)

    def predict_next_pos(self):
        raise NotImplementedError

    def area_of_likely_appearance(self, time):
        def match_error(pos, radius):
            dist_from_last_pos = np.linalg.norm(np.array(pos) - np.array(self.pos_history[-1]))
            if dist_from_last_pos > 0.5:
                return np.inf
            return dist_from_last_pos
        return match_error

class Tracker(object):
    def __init__(self, args):
        self.args = args
        self.active_tracks = {}
        self.id_counter = IdCounter()
        # metadata for convenience
        self.latest_matches = []
        self.new_tracks = []

    def match(self, cogs, radii, time):
        """ 
        inputs: 
          cogs : C.O.G.s in fixed frame
          radii : radii of detections
          time : rospy.Time of scan from which detections were taken
        """
        unmatched_detections_idcs = set(range(len(cogs)))
        self.latest_matches = []
        self.new_tracks = []
        # look for latest match for each active track
        for trackid in self.active_tracks:
            track = self.active_tracks[trackid]
            bestmatchidx = None
            bestmatcherror = np.inf
            for i in unmatched_detections_idcs:
                cog, r = cogs[i], radii[i]
                matcherror = track.area_of_likely_appearance(time)(cog, r)
                if matcherror < bestmatcherror:
                    bestmatchidx = i
                    bestmatcherror = matcherror
            if bestmatchidx is not None:
                unmatched_detections_idcs.remove(bestmatchidx)
                cog, r = cogs[bestmatchidx], radii[bestmatchidx]
                track.add_detection(cog, r, time)
                self.latest_matches.append(trackid)
        # create new tracks for unmatched detections
        for i in unmatched_detections_idcs:
            newtrack = Track(self.id_counter.next_id())
            newtrack.add_detection(cogs[i], radii[i], time)
            self.active_tracks[newtrack.id] = newtrack
            self.new_tracks.append(newtrack.id)
        # drop old tracks
        to_delete = []
        for trackid in self.active_tracks:
            track = self.active_tracks[trackid]
            if (time - track.time_history[-1]) > rospy.Duration(10.):
                to_delete.append(trackid)
        for trackid in to_delete:
            del self.active_tracks[trackid]

    def vizualize_tracks(self):
        for trackid in self.active_tracks:
            track = self.active_tracks[trackid]
            xy = np.array(track.pos_history)
            if trackid in self.latest_matches:
                color = "green"
            elif trackid in self.new_tracks:
                color = "blue"
            else:
                color = "grey"
            plt.plot(xy[:,0], xy[:,1], color=color)
            patch = patches.Circle((xy[-1,0], xy[-1,1]), 1.*track.sum_radius/track.n_hits, facecolor=(1,1,1,0), edgecolor=color)
            plt.gca().add_artist(patch)



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
        self.msg_prev = None
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
        # Timers
        rospy.Timer(rospy.Duration(0.001), self.tf_callback)
        # data
        self.clusters_data = None
        self.tf_pastrobs_in_fix = []
        # visuals 
        self.fig = plt.figure("clusters")
        try:
            self.visuals_loop()
        except KeyboardInterrupt:
            print("Keyboard interrupt: shutting down.")
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

            # legs
            is_legs = [True if 0.03 < r and r < 0.15 and len(c) >= MIN_CLUSTER_SIZE else False for r, c in zip(radii, clusters)]

            self.clusters_data = (xx, yy, clusters, is_legs, cogs, radii, self.tf_rob_in_fix)

            leg_cogs_in_fix = []
            leg_radii = []
            for i in range(len(is_legs)):
                if is_legs[i]:
                    leg_cogs_in_fix.append(apply_tf(cogs[i], Pose2D(self.tf_rob_in_fix)))
                    leg_radii.append(radii[i])
            self.tracker.match(leg_cogs_in_fix, leg_radii, msg.header.stamp)

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
            print("DWA callback: {:.2f} Hz | C.O.Gs, Radii: {} ms, Clustering: {} ms".format(
                1/(atoc-atic), cog_radii_time*1000., clustering_time*1000.))

    def visuals_loop(self):
        while True:
            with self.lock:
                if self.clusters_data is None:
                    plt.pause(0.1)
                    rospy.timer.sleep(0.01)
                    continue
                xx, yy, clusters, is_legs, cogs, radii, tf_rob_in_fix = self.clusters_data
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
                    patch = patches.Circle(cog, r, facecolor=(0,0,0,0), edgecolor=(0,0,0,1), linewidth=3)
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
