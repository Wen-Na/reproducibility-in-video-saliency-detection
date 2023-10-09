# Reproducibility in Video Saliency Detection

This repository contains the source code that we used to evaluate the approaches in our work "Saliency Detection in Videos: A Reproducibility Study with a Focus on Educational Videos"

In this study we evaluated three video approaches: TASED-Net [1], ViNet [2], and HD2S [3].
These were assessed on four metrics: Pearson’s correlation coefficient (CC), similarity (SIM), normalized scanpath saliency (NSS), and Judd area under the curve  variant (AUC-J). The code presented in this repository implements these metrics.


## Video saliency detection approaches

The source code that implements each video saliency detection approach is provided by the authors and it is openly available:
- TASED-Net [1]: [github.com/MichiganCOG/TASED-Net](https://github.com/MichiganCOG/TASED-Net)
- ViNet [2]: [github.com/samyak0210/ViNet](https://github.com/samyak0210/ViNeta)
- HD2S [3]: [github.com/perceivelab/hd2s](https://github.com/perceivelab/hd2s)

## Datasets
We used three datasets openly available to the public:
- DHF1K [4]: [mm-cheng.net/videosal](https://mmcheng.net/videosal/)
- DIEM [5]: [facilitated by the ViNet repository](https://github.com/samyak0210/ViNet)
- Educational dataset [6] and [7]: [https://osf.io/m7gj4/](https://osf.io/m7gj4/) and [https://osf.io/ptj75/](https://osf.io/ptj75/)


## References

[1] Kyle Min and Jason J. Corso. Tased-net: Temporally-aggregating spatial encoder-decoder network for video saliency detection. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages 2394–2403. IEEE, 2019.

[2] Samyak Jain, Pradeep Yarlagadda, Shreyank Jyoti, Shyamgopal Karthik, Ramanathan Subramanian, and Vineet Gandhi. Vinet: Pushing the limits of visual modality for audio-visual saliency prediction. In IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2021, Prague, Czech Republic, September 27 - Oct. 1, 2021, pages 3520–3527. IEEE, 2021.

[3] Giovanni Bellitto, Federica Proietto Salanitri, Simone Palazzo, Francesco Rundo, Daniela Giordano, and Concetto Spampinato. Hierarchical domain- adapted feature learning for video saliency prediction. Int. J. Comput. Vis., 129(12):3216–3232, 2021.

[4] Wenguan Wang, Jianbing Shen, Fang Guo, Ming-Ming Cheng, and Ali Borji. Revisiting video saliency: A large-scale benchmark and a new model. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 4894–4903. Computer Vision Foundation/IEEE Computer Society, 2018.

[5] Parag K. Mital, Tim J. Smith, Robin L. Hill, and John M. Henderson. Clustering of gaze during dynamic scene viewing is predicted by motion. Cogn. Comput., 3(1):5–24, 2011.

[6] Jens Madsen, Sara U. J ́ulio, Pawel J. Gucik, Richard Steinberg, and Lucas C. Parra. Synchronized eye movements predict test scores in online video education. Proceedings of the National Academy of Sciences, 118(5):e2016980118, 2021.

[7] H. Zhang, K. M. Miller, X. Sun, and K. S. Cortina. Wandering eyes: eye movements during mind wandering in video lectures. Applied Cognitive Psychology, 34:449–464, 2020.


