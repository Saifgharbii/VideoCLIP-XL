import os
from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder


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
        fr = fr[:,:,::-1]
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims(normalize(fr), axis=(0, 1))
        vid_tube.append(fr) 
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube)
    
    return vid_tube


videoclip_xl = VideoCLIP_XL()
state_dict = torch.load("./VideoCLIP-XL.bin", map_location="cpu")
videoclip_xl.load_state_dict(state_dict)
videoclip_xl.cpu().eval()

       
videos = [
    "./testing_fail.mp4",
]

texts = [
    "The video begins with a close-up shot of a white mug being filled with a dark brown liquid, presumably hot chocolate. The mug has a playful black and white drawing on its side. As the mug is filled, two dollops of white, fluffy whipped cream are added to the top, and then a fine powder is sprinkled over the cream, creating a delightful contrast.",
    "football match in a stadium",
    "a man making hot chocolate in a kitchen",
    "a woman playing a piano in a living room",
    "The video shows a person in a kitchen setting, adding ingredients into a small metal saucepan placed on an induction cooktop. These ingredients include what appears to be milk from a white bowl, followed by a smaller, measured amount of a white solid from a tiny silver cup, and then a light brown granular substance from another white bowl. The mixture is then whisked in the saucepan, turning into a smooth, light brown liquid. The whisked liquid is then transferred into a mug.",
    "A man rapidly exits a house and attempts to descend a wooden staircase. However, he quickly loses his footing and tumbles down the steps, landing ungracefully at the bottom before slowly getting up.",
    "A man hurries out of a house and falls down a flight of wooden stairs, then gets back up.",
    "A woman hurries out of a house and falls down a flight of wooden stairs, then gets back up.",
    "The man fell on a set of wooden stairs, which appear to lead from a house's exit down to a lower level, possibly a yard or pathway. The stairs are relatively wide and have a natural, unpainted wood finish, suggesting an outdoor or semi-outdoor setting.",
    "The woman fell on a set of wooden stairs, which appear to lead from a house's exit down to a lower level, possibly a yard or pathway. The stairs are relatively wide and have a natural, unpainted wood finish, suggesting an outdoor or semi-outdoor setting.",
    "A wooden staircase leading away from a house. The broad, unpainted timber steps suggest an outdoor transition area, likely connecting a building entrance to a ground-level pathway or garden."
]

with torch.no_grad():
    video_inputs = torch.cat([video_preprocessing(video) for video in videos], 0).float().cpu()
    video_features : torch.Tensor = videoclip_xl.vision_model.get_vid_features(video_inputs).float()
    video_features  = video_features / video_features.norm(dim=-1, keepdim=True)
    print(f"video_features shape: {video_features.shape}")

    text_inputs = text_encoder.tokenize(texts, truncate=True).cpu()
    text_features = videoclip_xl.text_model.encode_text(text_inputs).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f"text_features shape: {text_features.shape}")

Tmp = 100.

sim_matrix = (text_features @ video_features.T) * Tmp

print(sim_matrix)

videoclip_xl.text_model.state_dict()