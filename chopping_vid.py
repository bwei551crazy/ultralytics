"""
Extracts the frames from a given video source. 
"""

import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps = 1):
    """
    Args:
        video_path: path to video file used for extraction
        output_dir: directory to save extracted frames
        fps       : number of frames extracted per second. (default: 1 frame per second)

    """

    #creating output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok = True)

    #open vid file and obtain its properties
    video = cv2.VideoCapture(video_path)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    #calculate frame interval
    frame_interval = int(original_fps/fps)
    estimated_time = total_frames // frame_interval
    progress_bar = tqdm(total = estimated_time)
    
    print(f"frame_interval: {frame_interval}")

    save_count = 0

    for frame_count in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        #saving frame at the specified intervals
        if ((frame_count % frame_interval) == 0):
            frame_filename = os.path.join(output_dir, f"GZ_frame_{(save_count + 936):06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            save_count += 1
            progress_bar.update(1)
            
    progress_bar.close()
    video.release()
    print(f"Completed extraction. Saved {save_count} frames to {output_dir}")

if __name__ == "__main__":

    input_dir = "/home/yanjiaqi/own_ultralytics/ultralytics/data/images/Guangzhou twilight drive.mp4"
    output_dir = "/home/yanjiaqi/own_ultralytics/ultralytics/datasets/Custom_fineTune"
    extract_frames(input_dir, output_dir, fps = 0.5)


