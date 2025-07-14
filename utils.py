import cv2

def read_video(path):
    """
    Reads a video file and returns its frames as a list of images.

    Args:
        path (str): Path to the input video file.

    Returns:
        List[np.ndarray]: List of video frames in BGR format.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    """
    Saves a list of video frames to an output video file.

    Args:
        output_video_frames (List[np.ndarray]): List of frames to be saved.
        output_video_path (str): Path to the output video file.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()
