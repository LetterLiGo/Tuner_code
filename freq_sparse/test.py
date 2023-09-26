import os, glob
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
import torch, torchaudio
from datetime import datetime
now = '{}'.format(datetime.now().strftime("%m%d_%H%M"))
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())

import librosa
import librosa.display
import soundfile as sf
def plot(x, sr=16000):
    S = librosa.stft(x, n_fft=256)
    magnitude, phase = librosa.magphase(S)
    magnitude = np.log10(magnitude)
    librosa.display.specshow((magnitude), sr=sr, x_axis='time', y_axis='linear')

sampling_rate=16000

from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", use_auth_token="your token")
from pyannote.audio import Inference
inference = Inference(model, window="whole", device=device)

sim_cal = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def SV(args, opt_AE, ori_wav, start=0, end=4):
    acc=0; asr=0
    ori_wav_end = int(args.duration*sampling_rate)
    if end>ori_wav_end:
        end_final = end
    else:
        end_final=ori_wav_end
    emb_mix = torch.Tensor(inference.infer(torch.cat([ori_wav[:,:start], ori_wav[:,start:end] + opt_AE, ori_wav[:,end:end_final]], dim=1))).to(device)
    emb_tuner = torch.Tensor(inference.infer(opt_AE)).to(device)
    emb_ori = torch.Tensor(inference.infer(ori_wav[:int(sampling_rate*args.duration)])).to(device)
    score_ori = sim_cal(emb_ori,emb_mix).cpu().detach_().numpy()[0]
    score_tar = sim_cal(emb_tuner,emb_mix).cpu().detach_().numpy()[0]
    if score_ori > args.thres:
        acc = 1
    if score_tar > args.thres:
        asr =  1
    return score_ori, score_tar, acc, asr

def CSI(args, opt_AE, ori_wav, ori_voxprints, start=0, end=4):
    acc=0; asr=0
    score_ori_list=np.zeros(6)
    score_tar_list=np.zeros(6)
    ori_wav_end=int(4*sampling_rate)
    if end>ori_wav_end:
        end_final=end
    else:
        end_final=ori_wav_end

    emb_mix = torch.Tensor(inference.infer(torch.cat([ori_wav[:,:start], ori_wav[:,start:end] + opt_AE, ori_wav[:,end:end_final]], dim=1))).to(device)
    emb_tuner = torch.Tensor(inference.infer(opt_AE)).to(device)
    emb_ori = torch.Tensor(inference.infer(ori_wav[:int(sampling_rate*args.duration)])).to(device)
    for i, tmp_spker in enumerate(args.spker_list):
        if args.speaker == tmp_spker:
            score_ori_list[i]=sim_cal(emb_ori,emb_mix).cpu().detach_().numpy()[0]
            score_tar_list[i]=sim_cal(emb_tuner,emb_mix).cpu().detach_().numpy()[0]
        else:
            score_ori_list[i]=sim_cal(emb_ori,ori_voxprints[tmp_spker]).cpu().detach_().numpy()[0]
            score_tar_list[i]=sim_cal(emb_tuner,ori_voxprints[tmp_spker]).cpu().detach_().numpy()[0]
    max_ori_score=np.max(score_ori_list)
    idx_ori=np.argmax(score_ori_list)
    max_tar_score=np.max(score_tar_list)
    idx_tar=np.argmax(score_tar_list)
    if args.spker_list[idx_ori]==args.speaker:
        acc=1
    if args.spker_list[idx_tar]==args.speaker:
        asr=1
    
    return max_ori_score, max_tar_score, acc, asr

def OSI(args, opt_AE, ori_wav, ori_voxprints, start=0, end=4):
    acc=0; asr=0
    score_ori_list=np.zeros(6)
    score_tar_list=np.zeros(6)
    ori_wav_end=int(4*sampling_rate)
    if end>ori_wav_end:
        end_final=end
    else:
        end_final=ori_wav_end
    emb_mix = torch.Tensor(inference.infer(torch.cat([ori_wav[:,:start], ori_wav[:,start:end] + opt_AE, ori_wav[:,end:end_final]], dim=1))).to(device)
    emb_tuner = torch.Tensor(inference.infer(opt_AE)).to(device)
    emb_ori = torch.Tensor(inference.infer(ori_wav[:int(sampling_rate*args.duration)])).to(device)
    for i, tmp_spker in enumerate(args.spker_list):
        if args.speaker == tmp_spker:
            score_ori_list[i]=sim_cal(emb_ori,emb_mix).cpu().detach_().numpy()[0]
            score_tar_list[i]=sim_cal(emb_tuner,emb_mix).cpu().detach_().numpy()[0]
        else:
            score_ori_list[i]=sim_cal(emb_ori,ori_voxprints[tmp_spker]).cpu().detach_().numpy()[0]
            score_tar_list[i]=sim_cal(emb_tuner,ori_voxprints[tmp_spker]).cpu().detach_().numpy()[0]
    max_ori_score=np.max(score_ori_list)
    idx_ori=np.argmax(score_ori_list)
    max_tar_score=np.max(score_tar_list)
    idx_tar=np.argmax(score_tar_list)
    if args.spker_list[idx_ori]==args.speaker and max_ori_score>args.thres:
        acc=1
    if args.spker_list[idx_tar]==args.speaker and max_tar_score>args.thres:
        asr=1
    return max_ori_score, max_tar_score, acc, asr

def test(args, opt_AE, ori_voxprints):
    files=sorted(glob.glob(args.root_path + args.speaker + '/*/*.flac'))
    victim_all=[]
    idx=0
    acc_sv=0; asr_sv=0; score_tar_all_sv=0; score_ori_all_sv=0
    acc_csi=0; asr_csi=0; score_tar_all_csi=0; score_ori_all_csi=0
    acc_osi=0; asr_osi=0; score_tar_all_osi=0; score_ori_all_osi=0
    for _, wav in enumerate(files):
        y1,sr1=librosa.load(wav)
        if librosa.get_duration(y=y1, sr=sr1)>=4:
            victim_all.append(wav)
  
            current_victim=torchaudio.load(victim_all[idx])[0][:,:int(args.duration*sampling_rate)]
            current_victim=current_victim.to(device)
            current_victim_pad=torch.cat([current_victim,torch.zeros((1,int(1*sampling_rate)), device=device, requires_grad=False)],dim=1)
            current_AE=torchaudio.load(opt_AE)[0][:,:]
            current_AE=current_AE.to(device)
            
            # start=int(np.random.random_sample()*1*sampling_rate) # for the sake of experiment. we always emit attack when hearing sounds.
            start = 0
            end=int(start+args.tuner*sampling_rate)
            score_ori_sv,score_tar_sv,acc_current_sv,asr_current_sv=SV(args, current_AE, current_victim_pad, start=start,end=end)
            score_ori_csi,score_tar_csi,acc_current_csi,asr_current_csi=CSI(args, current_AE, current_victim_pad, ori_voxprints, start=start, end=end)
            score_ori_osi,score_tar_osi,acc_current_osi,asr_current_osi=OSI(args, current_AE, current_victim_pad, ori_voxprints, start=start, end=end)

            # count the acc asr score_tar score_ori
            acc_sv+=acc_current_sv; acc_csi+=acc_current_csi; acc_osi+=acc_current_osi
            asr_sv+=asr_current_sv; asr_csi+=asr_current_csi; asr_osi+=asr_current_osi
            score_tar_all_sv+=score_tar_sv; score_tar_all_csi+=score_tar_csi; score_tar_all_osi+=score_tar_osi
            score_ori_all_sv+=score_ori_sv; score_ori_all_csi+=score_ori_csi; score_ori_all_osi+=score_ori_osi

            idx += 1
            if idx>=20:
                break
    
    return (acc_sv/20, asr_sv/20, score_ori_all_sv/20, score_tar_all_sv/20), (acc_csi/20, asr_csi/20, score_ori_all_csi/20, score_tar_all_csi/20), (acc_osi/20, asr_osi/20, score_ori_all_osi/20, score_tar_all_osi/20)
