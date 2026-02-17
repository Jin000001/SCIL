## SCIL
In today's connected world, the generation of massive streaming data across diverse domains has become commonplace. In the presence of concept drift, class imbalance, label scarcity, and new class emergence, they jointly degrade representation stability, bias learning toward outdated distributions, and reduce the resilience and reliability of detection in dynamic environments. This paper proposes SCIL (Streaming Class-Incremental Learning) to address these challenges. The SCIL framework integrates an autoencoder (AE) with a multi-layer perceptron for multi-class prediction, uses a dual-loss strategy (classification and reconstruction) for prediction and new class detection, employs corrected pseudo-labels for online training, manages classes with queues, and applies oversampling to handle imbalance. The rationale behind the method's structure is elucidated through ablation studies and a comprehensive experimental evaluation is performed using both real-world and synthetic datasets that feature class imbalance, incremental classes, and concept drifts. Our results demonstrate that SCIL outperforms strong baselines and state-of-the-art methods. Based on our commitment to Open Science, we make our code and datasets available to the community.

## Paper
You can get a free copy of the accepted version from arXiv (https://arxiv.org/abs/2602.09681).

## Instructions
Please check the “instructions.txt” file.

## Requirements
Please check the “requirements.txt” file.

## Citation
If you have found our paper and/or part of our code and/or datasets useful, please cite our work as follows:
Li, Jin, Kleanthis Malialis, and Marios Polycarpou. "Resilient Class-Incremental Learning: on the Interplay of Drifting, Unlabelled and Imbalanced Data Streams." arXiv preprint arXiv:2602.09681 (2026).
