import numpy as np
import cv2
import torch
import dill
import hydra
import tqdm
import matplotlib.pyplot as plt
from glob import glob

from skill_utils.truncate import truncate
from skill_utils.format_pose import to_format, from_format
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)


class Model():
    def __init__(self, ckpt_path, computer):
        print("Started initialization.")
        self.payload=torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        print("Loaded checkpoint.")
        self.cfg=self.payload["cfg"]
        self.cls=hydra.utils.get_class(self.cfg._target_)
        self.workspace=self.cls(self.cfg)
        print("Loading payload...")
        self.workspace.load_payload(self.payload, exclude_keys=None, include_keys=None)
        
        print("Loading model...")
        if self.cfg.training.use_ema:
            self.policy=self.workspace.ema_model
        else:
            self.policy=self.workspace.model

        if computer=="remote":
            print("Moving model onto GPU...")
            self.device=torch.device('cuda')
        else:
            print("Using model on CPU.")
            self.device=torch.device("cpu")
        self.policy.eval().to(self.device)
        # set inference params
        self.policy.num_inference_steps = 16 # DDIM inference iterations
        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1

        self.obs_res=get_real_obs_resolution(self.cfg.task.shape_meta)
        print("Done.")

    def eval(self, obs):
        with torch.no_grad():
            self.policy.reset()
            obs_dict_np = get_real_obs_dict(
                env_obs=obs, shape_meta=self.cfg.task.shape_meta)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            result = self.policy.predict_action(obs_dict)
            action = result['action'][0].detach().to('cpu').numpy()
            
            del result
        return action
            

class Episode():
    def __init__(self, data_name_root):
        obs=np.load(data_name_root+"_obs.npy")
        actlog=np.load(data_name_root+"_act.npy", allow_pickle=True).item()
        act=[np.reshape(frame, (4,4)).T for frame in actlog["O_T_EE"]]
        self.obs,self.act=truncate(obs, act, T)

def load_model(computer):
    if computer=="remote":
        return Model("/home/romerur30/panda/diffusion_policy/data/outputs/2024.03.26/12.42.58_train_diffusion_unet_hybrid_line_image/checkpoints/latest.ckpt", computer)
    elif computer=="local":
        return Model("/root/skilllearn/model/latest.ckpt", computer)

def load_episode(computer):
    if computer=="remote":
        return Episode("/home/romerur30/panda/ag/data/21_03_2024_19_00_00")
    elif computer=="local":
        return Episode("/root/skilllearn/training_data/15_03_2024_12_54_44")

def eval_at_episode_index(model, episode, index):
    "eval model for observation at t=index of episode"
    # format one of the observations into the dict format accepted by the model
    obsdict={
        "agent_pos" : np.array([
            to_format(episode.act[index-1]),
            to_format(episode.act[index-2])]),
        "image" : np.array([
            episode.obs[index],
            episode.obs[index-1]]),
    }
    res=model.eval(obsdict)
    return res
    
def get_deltas_for_episode(model, episode):
    "evaluate at all indices and compare with expected response"
    results=[]
    deltas=[]
    for index in tqdm.tqdm(range(2,len(episode.obs))):
h        res=eval_at_episode_index(model, episode, index)
        results.append(res[0])
        deltas.append(res[0]-to_format(episode.act[index]))
    return results, deltas

def episode_delta_stats(model, episode):
    results, deltas=get_deltas_for_episode(model,episode)
    
    print(f"Min: {np.min(np.abs(deltas),0)}\nMax: {np.max(np.abs(deltas),0)}")

    norms=np.array([np.linalg.norm(i) for i in deltas])
    err_thres=0.3
    errcnt=len(norms[norms<0.3])
    print(f"Total err < {err_thres} count: {errcnt}")
    return errcnt
    

def get_episode_names(computer):
    if computer=="remote":
        return [i[:-8] for i in glob("/home/romerur30/panda/ag/data/*_act.npy")]
    else:
        raise NotImplementedError

def load_all_episodes(computer):
    epnames=get_episode_names(computer)
    return [Episode(i) for i in epnames]

# testing out what sort of response the model gives when prompted with training data

COMP="remote"
## INPUT
T=0.1
episode=load_episode(COMP)

## EVAL
model=load_model(COMP)

# episode_delta_stats(model,episode)

episodes=load_all_episodes(COMP)
errs=[]
for ep in episodes:
    errs.append(episode_delta_stats(model,ep))
print(errs)

# errs=[]
# for i in range(10):
#     errs.append(episode_delta_stats(model, episode))
# print(errs)    
## OUTPUT

