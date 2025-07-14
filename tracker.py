import os
import pickle
import torchreid
# import sys
import cv2
import numpy as np
# import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import supervision as sv
from ultralytics import YOLO
from collections import deque
from tqdm import tqdm

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.extractor = torchreid.utils.FeatureExtractor(model_name='osnet_ain_x0_25', device='cuda')
        self.reid_gallery = deque(maxlen=30)  # stores (global_id, embedding)
        self.global_id_counter = 0

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, device=0)
            detections += detections_batch
        return detections

    def get_object_tracks_with_embedding(self, frames,
                                         read_from_stub=False,
                                         stub_path=r"stubs/tracks_with_global_ids.pkl",
                                         gallery_stub_path=r"stubs/gallery.pkl"):

        tracks_with_global_ids = []

        def match_embedding_to_gallery(embedding, gallery, threshold):
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()

            best_score, best_id = -1, None
            gallery_dict = {}

            for gid, emb in gallery:
                if gid not in gallery_dict:
                    gallery_dict[gid] = []
                gallery_dict[gid].append(emb)

            for gid, embeds in gallery_dict.items():
                embeds_np = [e.detach().cpu().numpy() if hasattr(e, "detach") else e for e in embeds]
                avg_embed = np.mean(embeds_np, axis=0, keepdims=True)
                score = cosine_similarity(embedding.reshape(1, -1), avg_embed)[0][0]
                if score > best_score:
                    best_score = score
                    best_id = gid

            return best_id if best_score >= threshold else None

        # ──────── Load from stub if available ────────
        if read_from_stub and os.path.exists(stub_path) and os.path.exists(gallery_stub_path):
            print("Loading tracks and gallery from stubs...")
            with open(stub_path, 'rb') as f1, open(gallery_stub_path, 'rb') as f2:
                tracks_with_global_ids = pickle.load(f1)
                self.reid_gallery = pickle.load(f2)

                # Fix: extract current max global_id from the stored gallery
                ids = [gid for gid, _ in self.reid_gallery]
                self.global_id_counter = max(ids, default=-1) + 1

            return tracks_with_global_ids

        # ──────── Run detection + tracking + re-identification ────────
        detections = self.detect_frames(frames)

        for frame_num, detection in tqdm(enumerate(detections), total=len(detections), desc="Tracking + ReID"):
            frame = frames[frame_num]

            # Normalize class IDs
            boxes, scores, class_ids = [], [], []
            for i in range(len(detection.boxes)):
                cls = int(detection.boxes.cls[i])
                if cls in [1, 3]:  # goalkeeper or referee → player
                    cls = 2
                if cls not in [0, 2]:  # keep only ball and player
                    continue
                boxes.append(detection.boxes.xyxy[i].tolist())
                scores.append(float(detection.boxes.conf[i]))
                class_ids.append(cls)

            detection_supervision = sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(scores),
                class_id=np.array(class_ids)
            )

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            track_id_to_embedding = {}
            frame_tracks = []

            frame_buffer = []

            for det in detections_with_tracks:
                bbox = det[0].tolist()
                track_id = det[4]

                if track_id in track_id_to_embedding:
                    emb = track_id_to_embedding[track_id]
                else:
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    emb = self.extractor([crop])[0]
                    track_id_to_embedding[track_id] = emb

                matched_id = match_embedding_to_gallery(emb, self.reid_gallery, threshold=0.77)
                if matched_id is None:
                    matched_id = self.global_id_counter
                    self.global_id_counter += 1
                    print(f"[NEW ID] Assigned global_id: {matched_id}")


                frame_tracks.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "global_id": matched_id
                })
                frame_buffer.append((matched_id, emb))

            for gid, emb in frame_buffer:
                self.reid_gallery.append((gid, emb))
            
            tracks_with_global_ids.append(frame_tracks)

        # ──────── Save both tracks and gallery ────────
        if stub_path:
            with open(stub_path, 'wb') as f1, open(gallery_stub_path, 'wb') as f2:
                pickle.dump(tracks_with_global_ids, f1)
                pickle.dump(self.reid_gallery, f2)

        return tracks_with_global_ids

    def draw_tracks_with_global_ids(self, frames, tracks_with_global_ids, show_track_id=False):
        output_frames = []

        for i, frame in enumerate(frames):
            frame = frame.copy()
            for obj in tracks_with_global_ids[i]:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                global_id = obj['global_id']
                label = f'GID: {global_id}'

                if show_track_id and 'track_id' in obj:
                    label += f' (LID: {obj["track_id"]})'

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_frames.append(frame)

        return output_frames
