# Detailed description of **Harmonic Attention**

## Harmonic structure in speech
Spectrogram is a commonly used feature in speech enhancement methods, The following figure is an example (16K sample rate speech, 320 window lengths, 160 frame shifts).

![spectrogram](https://picx.zhimg.com/80/v2-00385054f8e4709ba3ef9262a65452b0_720w.jpeg?source%3Dd16d100b)

Selecting one of the frames for visualization can yield the following image.

![frame](https://pica.zhimg.com/80/v2-b3072a7d3973b350cc690466dd3517e5_720w.jpeg?source%3Dd16d100b)


It can be clearly seen that speech (voiced, vocal cord vibration) is presented on the spectrogram as regular comb-like structure, i.e., harmonic structure.

## The role of harmonic in speech enhancement
One outstanding solution proposed by Microsoft in 2019, PHASEN [1], stated that the speech enhancement model will actively try to fit the harmonic structure of the speech during the training process, as shown in the figure below.

![PHASEN](https://pica.zhimg.com/80/v2-9795732ef3fec9c35c0a5de43bb0901d_720w.png?source%3Dd16d100b)

It can be seen that the model's attention (denoted as Learned FTM weights) is similar to the manually set harmonic correlations (denoted as H=5, H=9, respectively), which proves that the harmonic structure of speech plays an important role in the enhancement task. This is because the harmonic, by virtue of its high-energy nature, is difficult to mask by noise across the frequency band, as shown in the following figure.

![mask](https://picx.zhimg.com/80/v2-33b7daab5420c2da40158401dc09f26b_720w.png?source%3Dd16d100b)

It is very clear from this example that while the noise will mask the speech signal over a large area, there are still certain harmonic structures that have not been completely masked. Also, since the deep learning model is trained based on gradient, it will preferentially fit the higher energy and more robust (salient) harmonic structure.

The position of harmonic distribution is determined by the vibration period of the vocal folds (pitch/fundamental frequency), so the corresponding harmonic position can be derived by presetting the fundamental frequency candidates (the common fundamental frequency range of human voice) as follows

![matrix](https://picx.zhimg.com/80/v2-446dd0edb316e9ab6f5b6c49e962930e_720w.png?source%3Dd16d100b)

The complete harmonic distribution is derived from the energy distribution of the spectrogram using the comb-pitch conversation matrix as shown below.

![compute](https://pica.zhimg.com/80/v2-8a5d2e2941718614b852fe0cd3a08824_720w.jpeg?source%3Dd16d100b)

The red distribution in the figure clearly fits the current frame signal better than the green distribution, and this process (the process of calculating the fit/correlation) can be simply represented as the integration of conversation matrix $\bm{U} \in \mathbb{R}^{N_c \times F}$ and spectrogram $\bm{S} \in \mathbb{R}^{F \times T}$ : $\bm{Q} =\bm{U} \cdot \bm{S} $. 
## Harmonic Attention
To ensure that the harmonic distribution (differentiable and trainable) is explicitly learned in the model, our previous work [2] proposed a Harmonic Attention mechanism, which consists of three stacked modules, i.e., convolution, harmonic integration, and frequency-channel recombination (FCR), as shown in the following figure.

![harmonic_attention](https://pic1.zhimg.com/80/v2-a2c8df7795b26941ba8b106ca99f077b_720w.png?source%3Dd16d100b)
### Harmonic Integration

![HA](https://picx.zhimg.com/80/v2-8a728d8fb3afb9907cad049589e327b3_720w.png?source%3Dd16d100b)

Harmonic integration is the most important part of Harmonic Attention, used to enable the model to actively capture harmonic and make structural corrections. The formulaic modeling of harmonic integration module is as follows，
1. Use convolutional modules to process input features to adapt to the processing of the conversation matrix $\bm{Q}$, $\bm{K}$ is obtained:

    $\bm{K} = \text{conv}(\text{norm}(\bm{X}_{\text{in}}^{2}))$

2. Integrate the adjusted $\bm{K}$ with the conversion matrix $\bm{Q}$ to calculate the correlation between each frame feature and the harmonic distribution corresponding to different fundamental frequencies.

    $\bm{sig} = \bm{K} \cdot \bm{Q}^\top$

3. In order to make it differentiable, the Softmax operation is used instead,


    $\bm{H} = \text{softmax}(\bm{sig}) \cdot \bm{Q}$

4. The selected harmonic distribution $\bm{Q}$ is used to guide the model in correction of harmonic structure.

    $\bm{V} = \text{conv}(\bm{X}_{in})$

    $\bm{X}_\text{out} = \text{conv}\left(\bm{V} \odot \text{conv}(\bm{H})\right)$

    $\text{conv}$ and $\text{norm}$ represent convolution and frequency domain layer normalization, respectively. 

Overall, the significance spectrum $\bm{sig}$ is obtained by integrating the conversion matrix $\bm{Q}$ with the normalized spectral features. The value of the significance spectrum represents the probability that the candidate fundamental frequency is the true fundamental frequency. The significance spectrum obtained through softmax is then multiplied with the conversion matrix (selecting the corresponding harmonic distribution based on confidence) to obtain the captured harmonic distribution spectrum $\bm{H}$. Finally, the predicted harmonic distribution is used to adjust the input spectrum features.

### frequency-channel Recombination

This model is composed of two stacked multi-head self-attention, one along with the channel (C-Attention) and the other one along with the frequency (F-Attention), to enhance the model's ability to communicate frequency domain and channel information.

![FCR](https://pica.zhimg.com/80/v2-7bc6886b294c91dd475ca27b47c5aab9_720w.png?source%3Dd16d100b)

### Temporal Modules
In the experiment, we tested two time-domain modeling strategies, LSTM-based DPRNN and Self-Attention. Please refer to the original text for detailed structure and processing.




```txt
[1] Yin, D., Luo, C., Xiong, Z., & Zeng, W. (2020, April). Phasen: A phase-and-harmonics-aware speech enhancement network. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 05, pp. 9458-9465).
[2] Wang, T., Zhu, W., Gao, Y., Zhang, S., & Feng, J. (2023). Harmonic Attention for Monaural Speech Enhancement. IEEE/ACM Transactions on Audio, Speech, and Language Processing.
```