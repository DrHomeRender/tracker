import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter

def iou(b1, b2):
    x1, y1, x2, y2 = np.maximum(b1[0], b2[0]), np.maximum(b1[1], b2[1]), np.minimum(b1[2], b2[2]), np.minimum(b1[3], b2[3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-6)

class Track:
    def __init__(self, mean, cov, track_id):
        self.mean, self.cov = mean, cov
        self.track_id = track_id
        self.time_since_update = 0
        self.hits = 1
        self.state = 'Tracked'

    def to_tlbr(self):
        x, y, a, h = self.mean[:4]
        w = a * h
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])

class BYTETracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, buffer_size=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.buffer_size = buffer_size
        self.tracks = []
        self.kf = KalmanFilter()
        self.next_id = 1

    def update(self, detections):
        # detections: [x1,y1,x2,y2,score]
        dets = detections[detections[:, 4] > self.track_thresh]
        track_boxes = np.array([t.to_tlbr() for t in self.tracks]) if self.tracks else np.zeros((0,4))
        iou_matrix = np.zeros((len(self.tracks), len(dets)))

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(dets):
                iou_matrix[i, j] = iou(t.to_tlbr(), d[:4])

        matched, unmatched_tracks, unmatched_dets = [], [], []
        if len(self.tracks) and len(dets):
            row, col = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row, col):
                if iou_matrix[r, c] >= self.match_thresh:
                    matched.append((r, c))
                else:
                    unmatched_tracks.append(r)
                    unmatched_dets.append(c)
        else:
            unmatched_dets = list(range(len(dets)))

        # Update matched tracks
        for r, c in matched:
            t = self.tracks[r]
            m = self.xyxy_to_xyah(dets[c, :4])
            t.mean, t.cov = self.kf.update(t.mean, t.cov, m)
            t.hits += 1
            t.time_since_update = 0
            t.state = 'Tracked'

        # Mark unmatched tracks
        for i, t in enumerate(self.tracks):
            if i not in [r for r, _ in matched]:
                t.time_since_update += 1
                if t.time_since_update > self.buffer_size:
                    t.state = 'Removed'

        # Create new tracks
        for idx in unmatched_dets:
            det = dets[idx]
            m = self.xyxy_to_xyah(det[:4])
            mean, cov = self.kf.initiate(m)
            self.tracks.append(Track(mean, cov, self.next_id))
            self.next_id += 1

        # Keep active
        self.tracks = [t for t in self.tracks if t.state != 'Removed']
        return [(t.track_id, *t.to_tlbr()) for t in self.tracks]

    @staticmethod
    def xyxy_to_xyah(bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x_c, y_c = x1 + w/2, y1 + h/2
        return np.array([x_c, y_c, w / np.maximum(h, 1e-6), h])
