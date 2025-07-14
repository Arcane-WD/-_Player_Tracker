
import os
import pickle
import sys
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from utils import get_bbox_width, get_center_of_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, device=0)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=r"stubs\tracks.pkl"):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            frame_tracks = []
            for det in detections_with_tracks:
                bbox = det[0].tolist()
                track_id = det[4]
                frame_tracks.append({"track_id": track_id, "bbox": bbox})
            tracks.append(frame_tracks)

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_tracks(self, frames, tracks):
        output_frames = []
        for i, frame in enumerate(frames):
            frame = frame.copy()
            for obj in tracks[i]:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {obj["track_id"]}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            output_frames.append(frame)
        return output_frames
