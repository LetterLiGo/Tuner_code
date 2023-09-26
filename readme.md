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