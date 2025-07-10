from utils import read_video, save_video
import cv2
import numpy as np
from ultralytics import YOLO
import torch
def main():
    video_frames = read_video(r"videos/input/inputvid.mp4")

    # Run video frames through model on gpu
    model = YOLO(r"model/best.pt")
    model.to(torch.device('cuda'))

    output_video_frames = []
    for i, frames in enumerate(video_frames):
        result = model.predict(frames, device='cuda', verbose=False)[0]
        output_frame = result.plot()
        output_video_frames.append(output_frame)
        print(f"Frame {i+1} out of {len(video_frames)}")

    #Save Video Frames
    save_video(output_video_frames, "videos\output\outputvid.avi")
if __name__ == "__main__":
    main()