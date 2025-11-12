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
        # Step 1: Predict all existing tracks
        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
        
        # Step 2: Filter high confidence detections
        dets = detections[detections[:, 4] > self.track_thresh] if len(detections) > 0 else np.zeros((0, 5))
        
        # Step 3: Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(dets)))
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(dets):
                iou_matrix[i, j] = iou(t.to_tlbr(), d[:4])

        # Step 4: Hungarian matching
        matched, unmatched_tracks, unmatched_dets = [], [], []
        if len(self.tracks) > 0 and len(dets) > 0:
            row, col = linear_sum_assignment(-iou_matrix)
            matched_indices = set()
            
            for r, c in zip(row, col):
                if iou_matrix[r, c] >= self.match_thresh:
                    matched.append((r, c))
                    matched_indices.add((r, c))
                    
            # Find unmatched tracks and detections
            matched_tracks = set(r for r, _ in matched)
            matched_dets = set(c for _, c in matched)
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
            unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]
        else:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(dets)))
        # Step 4: Hungarian matching

        # Step 4.5: Low-score detection second association (ByteTrack 핵심)
        low_dets = detections[(detections[:, 4] <= self.track_thresh) & (detections[:, 4] > 0.1)]
        if len(low_dets) > 0 and len(unmatched_tracks) > 0:
            iou_low = np.zeros((len(unmatched_tracks), len(low_dets)))
            for i, ut in enumerate(unmatched_tracks):
                for j, ld in enumerate(low_dets):
                    iou_low[i, j] = iou(self.tracks[ut].to_tlbr(), ld[:4])

            row, col = linear_sum_assignment(-iou_low)
            for r, c in zip(row, col):
                if iou_low[r, c] >= self.match_thresh * 0.9:  # 낮은 점수는 기준 완화
                    t = self.tracks[unmatched_tracks[r]]
                    m = self.xyxy_to_xyah(low_dets[c, :4])
                    t.mean, t.cov = self.kf.update(t.mean, t.cov, m)
                    t.hits += 1
                    t.time_since_update = 0
                    t.state = 'Tracked'

        # Step 5: Update matched tracks
        for r, c in matched:
            t = self.tracks[r]
            m = self.xyxy_to_xyah(dets[c, :4])
            t.mean, t.cov = self.kf.update(t.mean, t.cov, m)
            t.hits += 1
            t.time_since_update = 0
            t.state = 'Tracked'

        # Step 6: Mark unmatched tracks
        for i in unmatched_tracks:
            self.tracks[i].time_since_update += 1
            if self.tracks[i].time_since_update > self.buffer_size:
                self.tracks[i].state = 'Removed'

        # Step 7: Create new tracks from unmatched detections
        for idx in unmatched_dets:
            det = dets[idx]
            m = self.xyxy_to_xyah(det[:4])
            mean, cov = self.kf.initiate(m)
            self.tracks.append(Track(mean, cov, self.next_id))
            self.next_id += 1

        # Step 8: Remove dead tracks
        self.tracks = [t for t in self.tracks if t.state != 'Removed']
        return [(t.track_id, *t.to_tlbr()) for t in self.tracks]

    @staticmethod
    def xyxy_to_xyah(bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x_c, y_c = x1 + w/2, y1 + h/2
        return np.array([x_c, y_c, w / np.maximum(h, 1e-6), h])
