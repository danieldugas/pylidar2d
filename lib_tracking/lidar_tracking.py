import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import rospy
from timeit import default_timer as timer

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

    def estimate_velocity(self):
        current_vel_estimate = np.array([0., 0.])
        # default estimate
        if len(self.time_history) < 2:
            return current_vel_estimate
        # get average velocity over last 2 seconds
        two_sec_pos_history = []
        two_sec_time_history = []
        MAX_PAST = rospy.Duration(2.)
        latest_time = self.time_history[-1]
        for pos, time in zip(self.pos_history, self.time_history):
            if (latest_time - time) > MAX_PAST:
                continue
            two_sec_pos_history.append(pos)
            two_sec_time_history.append(time.to_sec())
        # instantaneous velocities at each point in history
        pos_deltas = np.diff(two_sec_pos_history, axis=0)
        t_deltas = np.diff(two_sec_time_history, axis=0)
        instant_velocities = pos_deltas / t_deltas[:, None]
        current_vel_estimate = np.mean(instant_velocities, axis=0)
        return current_vel_estimate

    def avg_radius(self):
        if self.n_hits == 0:
            return None
        return 1. * self.sum_radius / self.n_hits

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
        trackid_for_each_detection = [None for _ in cogs]
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
                trackid_for_each_detection[bestmatchidx] = trackid
        # create new tracks for unmatched detections
        for i in unmatched_detections_idcs:
            newtrack = Track(self.id_counter.next_id())
            newtrack.add_detection(cogs[i], radii[i], time)
            self.active_tracks[newtrack.id] = newtrack
            self.new_tracks.append(newtrack.id)
            trackid_for_each_detection[i] = newtrack.id
        # drop old tracks
        to_delete = []
        for trackid in self.active_tracks:
            track = self.active_tracks[trackid]
            if (time - track.time_history[-1]) > rospy.Duration(1.):
                to_delete.append(trackid)
        for trackid in to_delete:
            del self.active_tracks[trackid]
        return trackid_for_each_detection

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
            plt.plot(xy[:,0], xy[:,1], linestyle='--', color=color, zorder=0)
            patch = patches.Circle((xy[-1,0], xy[-1,1]), 1.*track.sum_radius/track.n_hits, facecolor=(1,1,1,0), edgecolor=color, zorder=2)
            plt.gca().add_artist(patch)



