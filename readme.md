# Enrollment-stage Backdoor Attacks on Speaker Recognition Systems via Adversarial Ultrasound

This is the core implementation for (Tuner) "[Enrollment-stage Backdoor Attacks on Speaker Recognition Systems via Adversarial Ultrasound](https://arxiv.org/abs/2306.16022)", Published in Internet of Things Journal (SCI, IF:11.1).

## Citation
If you think this repo helps you, please consider cite in the following format.
```latex
@ARTICLE{li2023tuner,
  author={Li, Xinfeng and Ze, Junning and Yan, Chen and Cheng, Yushi and Ji, Xiaoyu and Xu, Wenyuan},
  journal={IEEE Internet of Things Journal}, 
  title={Enrollment-Stage Backdoor Attacks on Speaker Recognition Systems via Adversarial Ultrasound}, 
  year={2024},
  volume={11},
  number={8},
  pages={13108-13124},
  keywords={Ultrasonic imaging;Spectrogram;Tuners;Speaker recognition;Acoustics;Training;Microphones;Adversarial ultrasound;backdoor attack;enrollment;speaker recognition},
  doi={10.1109/JIOT.2023.3328253}}
```

# Requirements
```
pip install -r requirements.txt
```

# the folder for freq_sparse
Codes in this folder are used for optimizing the Tuner and test its performance.

1. Prepare the datasets.
2. With the "python train.py" operation, you shall get a Tuner audio saved in the result folder.
3. you can freely call the prepared functions in test.py, then you will get the OSI, CSI, SV results, respectively.


# the folder for freq_compensate
Codes in this folder are used for real-world scenarios.

After generating our noise, the following steps need to be performed in order:
1. Pre-compensate the noise with *frequency_domain_compensation.m*. We provide an estimated average compensation curve in this function. You can also use your own curve by simply replacing the *y* in the function
2. Modulate the noise to a high-frequency carrier with *modulate_and_store.m*. Then the output audio file can be played through a laptop with a soundcard sampling rate higher than 96 kHz
