# End-to-end environment sound classification using 1D CNN

## literature

- SB-CNN
  - augmented dataset by time stretching, pitch shifting, dynamic range compression, adding background noise
  - 79% accuracy with augmented data
- best results obtained using a VGG 2D CNN as feature extractor and SVMs as classifiers
  - mean accuracy of 70^
- Boddapati et al. (2017) used spectrogram, MFCC, Cross Recurrence Plot, AlexNet and GoogLeNet for classication of 2D representations of environmental sounds
  - accuracy of 92-93%
- Zeghidour et. al (2018) proposed 1D CNN for speech recognition by learning a filter bank which is considered as a replacement of Mel-filterbanks
- combining 2 CNNs that learn from raw audio and 2D representations can be combined
  - Li et al (2018) trained independently using RawNet (raw audio waveforms) and MelNet (log-Mel features)
    - combined using Dempster-Shafer method
    - 92.2% accuracy on UrbanSound8k
  - Su, Zhang et al (2019) proposed TSCNN-DS also combined using DS method
    - 5 features log-mel spectrogram (LM), MFCC, Chroma, Spectral contrast and Tonnetz (CST).
    - LM and CST stacked and considered 1 feature
    - MFCC and CST also combined by stacking
    - 2 feature sets used to train 2 identical 4-layer CNNs
    - prediction combined using DS method
    - 97.2% on UrbanSound8k dataset

## this paper

- does not require data augmentation or any signal pre-processing for extracting features
- 2 contributions:
  - end-to-end 1D CNN initialized with gammatone filterbanks that has few parameters and does not need large amount of data
  - can handle audio signals of any length using a sliding window of appropriate width that breaks up audio signal into short frames of dimension compatible with input layer of the end-to-end 1D CNN

### variable length audio

- split audio into several frames of fixed length using a sliding window that overlaps adjacent frames
- overlapping can be viewed as data augmentation
- 16kHz sampling rate

### 1D CNN topology

- 4 CONV + MAXPOOLING layers + 2 FC layers + output layer (using softmax activation)
- input dimension: 16000 (1s of audio sampled at 16kHz)
  - pad if smaller
- large receptive fields in first CONV layers as first layer should have a more global view of the audio signal
- batch normalization applied after activation fn of each CONV layer
- mean squared logarithmic error used as loss function
- 2 FC layers with 128 and 64 neurons with dropout with p=0.25 for both layers
- ReLU used for all except output layer
- CONV layers inspired from SoundNet

### gammatone filterbanks

### aggregation of audio frames

- if input X is split into X1, X2, X3,..., XN inputs, those inputs are combined using majority rule to get the final output for X



# sampling rate of training data

```console
$ for i in data/UrbanSound8K/audio/*/*.wav;do soxi $i|grep 'Sample Rate'|sed 's/Sample Rate.*://g';done|sort -n|uniq -c
     12  8000
      7  11024
     39  11025
     45  16000
     44  22050
     82  24000
      4  32000
   5370  44100
   2502  48000
    610  96000
     17  192000
```

# A Dataset and Taxonomy for Urban Sound Research

- salience: occurrence was perceived to be in foreground (1) or background (2)

# reference papers

- CRNN - <https://arxiv.org/pdf/1602.05875v3.pdf>
- <https://arxiv.org/pdf/1904.08990v1.pdf>
- 1D convolutional neural networks and applications: A survey
- A Dataset and Taxonomy for Urban Sound Research

# ideas to try

- ~~stride is set to 50% - try 25%, 75%, 100%~~
- try 16, 22.1 kHz sampling rate
- ~~batch_size vary 32, 64, 100~~
- try input normalization
- try changing dropout percent
- add l2 regularization
- architecture
  - add batchNormalization after CONV layers
  - add dropout layers after

# todo

- ignore chunk if size < 50% of chunk size
  - sizes: ((97925, 16000, 1), (97925, 10), (10449, 16000, 1), (10449, 10))
  - accuracy: 0.5722819593787336
- ignore chunk if size <= 100% of chunk size except if that is the only chunk
  - sizes: ((90453, 16000, 1), (90453, 10), (9646, 16000, 1), (9646, 10))
  - accuracy: 0.6057347670250897
- ignore chunk if size <= 100% of chunk size
  - sizes: ((90030, 16000, 1), (90030, 10), (9612, 16000, 1), (9612, 10))
  - accuracy: 0.6575342465753424
- overlap 0 25 50 75
  - 0
    - sizes: ((28274, 16000, 1), (28274, 10), (3008, 16000, 1), (3008, 10))
    - accuracy: 0.6093189964157706
  - 25
    - sizes: ((35162, 16000, 1), (35162, 10), (3746, 16000, 1), (3746, 10))
    - accuracy: 0.5997610513739546
  - 50
    - sizes: ((48983, 16000, 1), (48983, 10), (5218, 16000, 1), (5218, 10))
    - accuracy: 0.5902031063321386
  - 75
    - sizes: ((90453, 16000, 1), (90453, 10), (9646, 16000, 1), (9646, 10))
    - accuracy: 0.6618876941457587

- loss: cross entropy
  - accuracy: 0.5902031063321386
- optimizer = adam
  - accuracy: 0.6630824372759857
- batch_size:
  - default: 100
  - 32:
    - accuracy: 0.6212664277180406
  - 64:
    - accuracy: 0.5842293906810035
  - 100:
    - accuracy: 0.6152927120669056
  - 128:
    - accuracy: 0.6224611708482676
  - 256:
    - accuracy: 0.6296296296296297
- learning rate (adam):
  - default 0.6630824372759857
  - 0.0001: 0.6738351254480287, 0.6320191158900836
  - 0.0005: 0.6272401433691757
  - 0.00001: 0.6499402628434886
  - 0.00005: 0.6224611708482676
- shape 16000:1, 1:16000
  1, 16000 not possible - 16000, 1 is correct
- 2d: 224 image dimension
  - no chunks
  - original accuracy: 0.7995018679950187
  - 72x72 75% overlap: 0.6662515566625156, 0.6737235367372354
  - 224x224 original spectrogram: 0.7287933230400085
  - 128 dense layer: 0.6899128268991283
  - 64 dense layer: 0.6899128268991283
  - 1: 0.706875753920386
- old 1d accuracies: [0.5521191294387171,
 0.47184684684684686,
 0.43675675675675674,
 0.6525252525252525,
 0.6378205128205128,
 0.5504252733900364,
 0.6026252983293556,
 0.5,
 0.6004901960784313,
 0.6344086021505376]

# ensemble

0.7129071170084439 0.623642943305187 0.7008443908323281
0.7109756097560975 0.6048780487804878 0.723170731707317
0.5548022598870056 0.45084745762711864 0.5649717514124294
0.7547568710359408 0.6374207188160677 0.6670190274841438
0.7997737556561086 0.7081447963800905 0.7601809954751131
0.748062015503876 0.6201550387596899 0.7235142118863049
0.6945137157107232 0.6084788029925187 0.6508728179551122


- ~~3 folds left to train~~
- 0.1 to 0.9 weights
- do not write stuff that is not defendable


# results

- average accuracy of ensemble, 1D CNN, 2D CNN: (0.7237669426885855, 0.6098607853566553, 0.6974624641169974)
- stddev of ensemble, 1D CNN, 2D CNN: (0.06449183471232027, 0.07173799318149568, 0.05394596276264152)
- accuracies of 10 folds of ensemble, 1D CNN, 2D CNN:

  ```
  [(0.7129071170084439, 0.623642943305187, 0.7008443908323281),
   (0.7109756097560975, 0.6048780487804878, 0.723170731707317),
   (0.5548022598870056, 0.45084745762711864, 0.5649717514124294),
   (0.7547568710359408, 0.6374207188160677, 0.6670190274841438),
   (0.7997737556561086, 0.7081447963800905, 0.7601809954751131),
   (0.748062015503876, 0.6201550387596899, 0.7235142118863049),
   (0.6945137157107232, 0.6084788029925187, 0.6508728179551122),
   (0.7273936170212766, 0.5212765957446809, 0.7194148936170213),
   (0.7474358974358974, 0.6948717948717948, 0.7435897435897436),
   (0.7870485678704857, 0.6288916562889165, 0.7210460772104608)]
  ```
- varying weights:
  ```
  fold=0, max_w1=0.4, max_acc=0.7189384800965019
  fold=1, max_w1=0.30000000000000004, max_acc=0.7317073170731707
  fold=2, max_w1=0.1, max_acc=0.5683615819209039
  fold=3, max_w1=0.5, max_acc=0.7547568710359408
  fold=4, max_w1=0.4, max_acc=0.8031674208144797
  fold=5, max_w1=0.4, max_acc=0.748062015503876
  fold=6, max_w1=0.6, max_acc=0.699501246882793
  fold=7, max_w1=0.4, max_acc=0.7579787234042553
  fold=8, max_w1=0.4, max_acc=0.7538461538461538
  fold=9, max_w1=0.5, max_acc=0.7870485678704857
  ```
- weight = 0.4

  ```
  [0.7189384800965019,
   0.7304878048780488,
   0.5661016949152542,
   0.7367864693446089,
   0.8031674208144797,
   0.748062015503876,
   0.6571072319201995,
   0.7579787234042553,
   0.7538461538461538,
   0.7820672478206725]
  ```
- avg, stddev: (0.725454324254405, 0.06471490083079626)
