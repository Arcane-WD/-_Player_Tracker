import cv2
from tracker import Tracker
from utils import read_video, save_video

def main():
    """
    Main pipeline for object detection, tracking, and ReID-based global ID assignment.

    Steps:
    1. Read input video.
    2. Run detection + ByteTrack + ReID embedding association.
    3. Assign global identities to players and the ball.
    4. Draw global IDs on frames.
    5. Save annotated frames as an output video.
    """

    # Input/output paths
    video_path = 'videos/input/inputvid.mp4'         # Input football match video
    model_path = 'model/best.pt'                     # Path to trained YOLOv8 model
    output_path = 'videos/output/outputvid.avi'      # Path to save annotated video

    # Load video frames
    frames = read_video(video_path)

    # Initialize detection + tracking + ReID module
    tracker = Tracker(model_path)

    # Get per-frame object tracks with consistent global identities
    tracks = tracker.get_object_tracks_with_embedding(frames, read_from_stub=True)

    # Draw bounding boxes + global IDs on frames
    output_frames = tracker.draw_tracks_with_global_ids(frames, tracks)

    # Save annotated frames as video
    save_video(output_frames, output_path)

if __name__ == "__main__":
    main()
