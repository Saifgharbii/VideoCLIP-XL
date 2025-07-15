import torch
import cv2
import numpy as np
from time import time
from utils.vision_encoder import get_vision_encoder

model_path = "videoclip_xl_vision_encoder.pt"

vision_model = get_vision_encoder()
vision_model.load_state_dict(torch.load(model_path, map_location="cpu"))
vision_model.eval()

def _frame_from_video(video: cv2.VideoCapture)   :
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
        
        
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data / 255.0 - v_mean) / v_std
        
    
def video_preprocessing(video_path, fnum=8):
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]
    step = len(frames) // fnum
    print(f"Total frames: {len(frames)}, Step: {step}, Target frames: {fnum}")
    frames = frames[::step][:fnum]

    vid_tube = []
    for fr in frames:
        fr = fr[:,:,::-1] #BGR to RGB
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims(normalize(fr), axis=(0, 1))
        vid_tube.append(fr) 
    print(f"Frames shape: {vid_tube[0].shape}")
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3)) #(B, T, C, H, W) format
    print(f"Video shape: {vid_tube.shape}")
    vid_tube = torch.from_numpy(vid_tube)
    
    return vid_tube

def video_preprocessing_batch(video_path, fnum=8, batch_size=4):
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]
    total_frames = len(frames)
    print(f"Total frames: {total_frames}")

    total_clips = total_frames // fnum
    total_batches = total_clips // batch_size

    print(f"Total clips: {total_clips}, Total batches: {total_batches}")

    all_batches = []

    for b in range(total_batches):
        batch_clips = []
        for i in range(batch_size):
            start = (b * batch_size + i) * fnum
            clip = frames[start:start+fnum]
            processed_frames = []

            for fr in clip:
                fr = fr[:, :, ::-1]  # BGR to RGB
                fr = cv2.resize(fr, (224, 224))
                fr = normalize(fr)
                processed_frames.append(fr)

            clip_np = np.stack(processed_frames, axis=0)  # (T, H, W, C)
            clip_np = np.expand_dims(clip_np, axis=0)     # (1, T, H, W, C)
            batch_clips.append(clip_np)

        batch_np = np.concatenate(batch_clips, axis=0)     # (B, T, H, W, C)
        batch_np = np.transpose(batch_np, (0, 1, 4, 2, 3))  # (B, T, C, H, W)
        batch_tensor = torch.from_numpy(batch_np).float()
        all_batches.append(batch_tensor)

    return all_batches  # list of tensors

def video_only_embedder(video_path: str, fnum: int = 8) -> torch.Tensor:
    """
    Embed a video using the vision encoder.

    Args:
        video_path (str): Path to the video file.
        fnum (int): Number of frames to sample from the video.

    Returns:
        torch.Tensor: Embedded video tensor.
    """
    vid_tube = video_preprocessing_batch(video_path, fnum,1)[0].float().cpu()
    print(f"Video tube shape: {vid_tube.shape}")
    t1 = time()
    with torch.no_grad():
        vid_emb = vision_model.get_vid_features(vid_tube).float()
    t2 = time()
    print(f"Embedding time: {t2 - t1:.2f} seconds")
    return vid_emb

def video_only_embedder_traced(video_path: str, fnum: int = 8) -> torch.Tensor:
    """
    Embed a video using the vision encoder.

    Args:
        video_path (str): Path to the video file.
        fnum (int): Number of frames to sample from the video.

    Returns:
        torch.Tensor: Embedded video tensor.
    """
    vid_tube = video_preprocessing_batch(video_path, fnum,1)[0].float().cpu()
    vid_tube = vid_tube.permute(0, 2, 1, 3, 4).contiguous()
    print(f"Video tube shape: {vid_tube.shape}")
    vision_model = torch.jit.load("traced_vision_model.pt")
    vision_model.eval()
    print("Loaded traced model")
    t1 = time()
    with torch.no_grad():
        vid_emb = vision_model(vid_tube).float()
    t2 = time()
    print(f"Embedding time: {t2 - t1:.2f} seconds")
    return vid_emb

if __name__ == "__main__":
    video_path = "testing.mp4"
    video_embedding = video_only_embedder_traced(video_path)
    print(video_embedding.shape)  