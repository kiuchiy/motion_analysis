import numpy as np
from scipy.spatial import distance
from collections import Counter


class TrackHumans:
    def __init__(self, start_frame=0,):
        self.start = start_frame
        self.humans_id = None
        self.humans_current = None
        self.humans_tracklet = None
        self.clm_num = None
        self.humans_post = None

    def track_humans(self, frame, humans):
        """
        Make tracklets (tracked ids cluster).
        1. set current id by referring nearest id of previous frame
        2. add id to humans data
        3. stack humans data to make tracklet
        :param :
        :return:
        """
        # initialize
        humans_current = np.concatenate((np.c_[np.repeat(frame, len(humans))],
                                         humans.reshape(humans.shape[0], humans.shape[1] * humans.shape[2])), axis=1)
        if frame == self.start:
            self.humans_id = np.array(range(len(humans)))
            self.humans_current = np.concatenate((np.c_[self.humans_id], humans_current), axis=1)
            self.humans_tracklet = self.humans_current
            self.clm_num = self.humans_current.shape[1] - 1

        else:
            self.humans_id = self.search_nearest(humans, self.humans_post, self.humans_id)
            self.humans_current = np.concatenate((np.c_[self.humans_id], humans_current), axis=1)
            self.humans_tracklet = (np.concatenate((self.humans_tracklet[self.humans_tracklet[:, 0] > (frame - 30)],
                                                   self.humans_current)))  # .astype(int)

        self.humans_post = humans

    @staticmethod
    def search_nearest(humans, prev_humans, prev_id):
        """
        
        :param humans: 
        :param prev_humans: 
        :param prev_id: 
        :return: 
        """
        # calculate humans points distances
        # 1.distance of body parts position (like Nose = humans[:,0,:2])
        distances = np.array([distance.cdist(humans[:, i, :2], prev_humans[:, i, :2]) for i in range(humans.shape[1])])

        # 2. search nearest body
        # distance mean of each body parts as representative distance value
        dists_from_prevs = np.nanmean(distances, axis=0)
        # nearest_prev_num means previous frame's body's index ordered by distance from current ones
        nearest_prev_num = np.nanargmin(dists_from_prevs, axis=1)
        # sort previous ids
        current_id = prev_id[nearest_prev_num]

        # diff in 1 frame should be less than threshold pixels
        min_dists_from_prev = np.min(dists_from_prevs, axis=1)
        track_threshold = 500
        # track_threshold = np.mean(min_dists_from_prev) + np.std(min_dists_from_prev)  # 500

        new_appearance = np.where(min_dists_from_prev > track_threshold)[0]
        # check the duplication of nearest body num
        duplicate_num = [item for item, count in Counter(nearest_prev_num).items() if count > 1]
        if len(duplicate_num):
            for dup in duplicate_num:
                target_num = np.where(nearest_prev_num == dup)
                correct_idx = np.argmin(dists_from_prevs[target_num, dup])
                new_appearance = np.concatenate((new_appearance, np.delete(target_num, correct_idx))).astype('int')
            current_id[new_appearance] = range(max(prev_id) + 1, max(prev_id) + 1 + len(new_appearance))
        return current_id
