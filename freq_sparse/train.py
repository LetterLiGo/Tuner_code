import os, glob
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import torch, torchaudio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())

from datetime import datetime
now = '{}'.format(datetime.now().strftime("%m%d_%H%M"))

import math
import numpy as np
from tqdm.notebook import tqdm


import librosa
import librosa.display
import soundfile as sf
def plot(x, sr=16000):
    S = librosa.stft(x, n_fft=256)
    magnitude, phase = librosa.magphase(S)
    magnitude = np.log10(magnitude)
    librosa.display.specshow((magnitude), sr=sr, x_axis='time', y_axis='linear')# return normalize(magnitude)

sampling_rate=16000
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", use_auth_token="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")
from pyannote.audio import Inference
inference = Inference(model, window="whole", device=device)

def init_oriprints(args) -> (dict):
    ori_voxprints={}
    for id in tqdm(args.spker_list):
        ori_wav = glob.glob(args.root_path + id + '/*/*.flac')[0:5]
        SUM=torch.zeros([1,512], device=device)
    
        for w in ori_wav:
            wav = torchaudio.load(w)[0][:, :int(args.duration*sampling_rate)]
            SUM+=torch.Tensor(inference.infer(wav)).to(device)
        MEAN=SUM/len(ori_wav)
        ori_voxprints[id] = MEAN
    
    return ori_voxprints

#robust
import torch.optim as optim
import argparse
# ---------------------------------------------------------  0809 Optimization based on Adam -------------------------------------------------
import math

sim_cal = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
ori_AE = None

def loss_estimate(args, ori_wav, oriprint, A, K, w_tar=3, w_ori=1.2, Lfreq=10, Lts=20):
    opt_AE = craft_AE_white(ori_wav, A, K, Lfreq=Lfreq, Lts=Lts, tuner=args.tuner)
    opt_AE_norm = opt_AE / abs(opt_AE).max()
    emb = torch.Tensor(inference.infer(ori_wav.data + opt_AE_norm)).to(device)      
    emb_AE = torch.Tensor(inference.infer(opt_AE_norm)).to(device)
    score_ori = sim_cal(emb, oriprint)
    score_tar = sim_cal(emb, emb_AE)
    loss=3-w_tar*score_tar-w_ori*score_ori

    return loss, score_tar, score_ori 

def craft_AE_white(ori_wav, A, K, sr=16000, Lfreq=3, Lts=5, tuner=None):
    dur = ori_wav.shape[-1] // Lts # 优化为Lts段 各种频率
    left = ori_wav.shape[-1] % Lts
    n = int(ori_wav.shape[-1] / Lts)

    if left == 0: left = dur
    
    for i, step in enumerate(zip(A, K)):
        a, k = step
        if i == n:
            timestamp = torch.range(start=0, end=left-1, step=1, device=device).unsqueeze(0)   
        else:
            timestamp = torch.range(start=0, end=dur-1, step=1, device=device).unsqueeze(0)

        cur_frame = timestamp / sr
        tmp = 0
        for index in range(Lfreq):
            tmp += a[index]*torch.sin(k[index]*2*(math.pi)*cur_frame)
        
            window = torch.hann_window(tmp.shape[1], device=device) # hanning window for less leakage
            tmp = window * tmp
        if i == 0:
            origin = tmp
        else:
            origin = torch.cat([origin, tmp], dim=1)
    return origin

def loss_estimate_robust2(args, ori_wav, oriprint, A, K, w_tar, w_ori, Lfreq=100, Lts=20):
    score_ori_all=0; score_tar_all=0; norm1_all = 0
    opt_AE = craft_AE_white(ori_wav[:,:int(args.tuner*sampling_rate)], A, K, Lfreq=Lfreq, Lts=Lts, tuner=args.tuner)

    opt_AE_norm = torch.clamp(opt_AE, min=-1, max=1)
    ori_wav_end = int(4*sampling_rate)
    for ii in range(args.time_step):
        start=int(np.random.random_sample()*1*sampling_rate)
        end=int(start+args.tuner*sampling_rate)
        for jj in range(args.amp_step):
            k=np.random.random_sample()*1.5+0.5
            if end>ori_wav_end:
                end_final=end
            else:
                end_final=ori_wav_end
            emb=torch.Tensor(inference.infer(torch.cat([k*ori_wav[:,:start], k*ori_wav[:,start:end] + opt_AE_norm, k*ori_wav[:,end:end_final]], dim=1))).to(device)
            
            emb_AE = torch.Tensor(inference.infer(opt_AE_norm)).to(device)
            score_ori = sim_cal(emb, oriprint)
            score_tar = sim_cal(emb, emb_AE)
            norm1 = torch.mean(torch.norm(A, p=1, dim=0))
            score_ori_all+=score_ori
            score_tar_all+=score_tar
            norm1_all+=norm1
    loss=3-w_tar*score_tar_all-w_ori*score_ori_all + 5*norm1_all
    return loss, score_tar, score_ori


def tune_robust_attack_white(args, ori_wav, ori_voxprints:dict, Lfreq=100, Lts=20):
    '''
        @ tuner: Tuner's length
        @ Lfreq: e.g., initially 100 | Lts: default: 20
        @ epoch: iteration times
        @ white: you can quickly train an AE based the setting and test for transferability.
    '''
    print(f'cur_spker:{args.speaker}, current Lfreq:{Lfreq}, Lts:{Lts}, time_step:{args.time_step}, amp_step:{args.amp_step}, tuner:{args.tuner}')
    ori_wav = (ori_wav[:,0:int(args.tuner*sampling_rate)]).to(device) # trunk the original wav
    oriprint = ori_voxprints[args.speaker] # averaged enrolled voiceprint
    ori_pad = torch.cat([ori_wav, torch.zeros((1,int(args.start_range*sampling_rate)), device=device, requires_grad=False)],dim=1)
    A = torch.full((Lfreq, Lts), ori_wav.max(), device=device).reshape(-1,Lfreq) # equal amplitude initialization
    A.requires_grad=True
    K = 4000*torch.rand(Lfreq*Lts, device=device).reshape(-1,Lfreq)
    K.requires_grad=False
    ori_AE = craft_AE_white(ori_wav, A, K, Lfreq=Lfreq, Lts=Lts, tuner=args.tuner) # init the ori_AE, for comparing with the opt_AE result
    sf.write(f'./result/tmp/4992/{args.tuner}/tuner-test.wav', librosa.util.normalize(ori_AE[0].cpu().detach_().numpy()), 16000)
    best_loss = 100
    opt1 = optim.Adam([A], lr=params['lr'])

    w_tar = params['w_tar']; w_ori = params['w_ori']
    for i in (range(args.epoch)):
        loss, _, _ = loss_estimate_robust2(args, ori_pad, oriprint, A, K, w_tar=w_tar,w_ori=w_ori,Lfreq=Lfreq, Lts=Lts)
        opt1.zero_grad()
        loss.backward(retain_graph=True)
        opt1.step()
        A_modified = torch.clamp(A, min=0, max=1)
        A_modified[A_modified<0.03] = 0
        A.data = A_modified.data
        
        if i % 10 == 0:
            loss_look, score_tar, score_ori = loss_estimate(args, ori_wav, oriprint, A, K, w_tar=w_tar,w_ori=w_ori, Lfreq=Lfreq, Lts=Lts)
            print(f"{i} {loss_look.detach().cpu().numpy()} score_tar:{score_tar.detach().cpu().numpy()} score_ori:{score_ori.detach().cpu().numpy()}")
    
    opt_AE = craft_AE_white(ori_wav, A, K, Lfreq=Lfreq, Lts=Lts, tuner=args.tuner)

    return ori_AE, opt_AE[0].cpu().detach_().numpy(), score_ori.cpu().detach_().numpy()[0], score_tar.cpu().detach_().numpy()[0]


import numpy as np
from datetime import datetime
import soundfile as sf
def train(args, ori_voxprints) -> (str):
    exp_path=f'./result/tmp/{args.speaker}/'+str(args.tuner)
    os.makedirs(exp_path, exist_ok=True)
    time_idx = '{}'.format(datetime.now().strftime("%m%d_%H%M"))

    wav_list = glob.glob(args.root_path + args.speaker + '/*/*.flac')
    for wav in wav_list:
        if sf.info(wav).duration >= args.duration:
            print('selected wav file:', wav)
            ori_wav = torchaudio.load(wav)[0][:, :int(args.duration*sampling_rate)]
            break

    ori_AE, final_AE, score_all_ori, score_all_tar = tune_robust_attack_white(args, ori_wav, ori_voxprints, Lts=params['lts'])
    save_AE = exp_path+'/tuner-'+str(args.tuner)+'-'+str(time_idx)+'-tar-'+str(round(score_all_tar,3))+'-ori-'+str(round(score_all_ori,3))+'.wav'
    sf.write(save_AE, librosa.util.normalize(final_AE), sampling_rate)

    return save_AE

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tuning')
    parser.add_argument('--tuner', type=float, default=3, help='Tuner duration')
    parser.add_argument('--epoch', type=int, default=200, help='optimization epoch numbers')
    parser.add_argument('--speaker', type=str, default="4992", help='targeted speaker')
    parser.add_argument('--root_path', type=str, default="./data/LibriSpeech/test-clean/", help='<your path to dataset>')
    parser.add_argument('--spker_list', type=list, default=['1995','6829','4992','6930','2830','5105'], help='The evaluation set')
    parser.add_argument('--duration', type=int, default=4, help='the speech length for enrollment')
    parser.add_argument('--thres', type=float, default=0.291)
    # ---------------- robust training ------------------
    parser.add_argument('--time_step', type=int, default=1, help='time augmentation') # small number for quick optimization
    parser.add_argument('--start_range', type=int, default=1, help='mimic the reaction time of the attacker')
    parser.add_argument('--amp_step', type=int, default=1, help='amplitude augmentation') # small number for quick optimization

    args = parser.parse_args()

    params = {'lts': 20, 'w_tar': 4, 'w_ori': 1.2, 'lr': 0.05, 'sigma': 0.2}

    ori_voxprints = init_oriprints(args)
    result_AE = train(args, ori_voxprints)


