import cv2
from tracker import Tracker
from utils import read_video, save_video

def main():
    video_path = 'videos/input/inputvid.mp4' 
    model_path = 'model/best.pt'  
    output_path = 'videos/output/outputvid.avi'

    frames = read_video(video_path)
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks_with_embedding(frames, read_from_stub=True)

    output_frames = tracker.draw_tracks_with_global_ids(frames, tracks)
    save_video(output_frames, output_path)

if __name__ == "__main__":
    main()
