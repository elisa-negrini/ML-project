# Cross-Domain Image Retrieval Project

This repository contains the models and associated code for our cross-domain image retrieval project, focused on matching real photographs with AI-generated images.

---

## Introduction

This project stemmed from a competition focused on cross-domain image retrieval, specifically matching real photographs of celebrities (queries) with synthetic AI-generated images (gallery) of the same individuals rendered in different visual styles. The challenge required developing robust feature extraction and similarity matching techniques to bridge the domain gap between natural and synthetic imagery.

The dataset consisted of approximately 5,000 training images organized by celebrity identity, with each subfolder containing both natural and synthetic images of the same person. The test set included around 1,500 query images (real photographs) and 1,500 gallery images (AI-generated synthetic images). For each query image, the task was to retrieve the 10 most similar gallery images, ranked by similarity.

The evaluation was based on three metrics contributing to a maximum score of 1,000 points: **Top-1 Accuracy** (600 points) which checks the correctness of the most similar retrieved image, **Top-5 Accuracy** (300 points) requiring at least one correct match within the top 5 results, and **Top-10 Accuracy** (100 points) requiring at least one correct match within the top 10 results.

### Overview of Approaches

Our methodology involved experimenting with three distinct deep learning architectures, each leveraging different aspects of visual feature extraction. Prior to the actual competition, we conducted preliminary experiments on three different datasets to validate our approaches: `images_clothes` focusing on the clothing domain with approximately 10,000 test images, `images_fish` on fish species with around 13,000 test images, and `images_animals` on the animal domain comprising roughly 4,000 training images, 800 query images, and 800 gallery images. These preliminary tests served as proof-of-concept evaluations before applying our methods to the celebrity face dataset.

#### ResNet-50 with Fine-tuning

We began with a **ResNet-50** architecture pre-trained on ImageNet, applying fine-tuning on the training dataset. The model was modified with a custom classification head to learn identity-specific features. This approach utilized standard preprocessing techniques including resizing, center cropping, and normalization during both training and inference. The model was developed and tested before the competition as part of our initial exploration of traditional computer vision approaches.

#### CLIP with Advanced Training Strategies

The second approach employed **CLIP (ViT-L-14)** with sophisticated training enhancements, also developed prior to the competition. We created a custom architecture combining **GeM (Generalized Mean) Pooling** for more effective aggregation of patch features from the Vision Transformer, **ArcFace Loss** to learn more discriminative embeddings with angular margin constraints, **Center Loss** to increase intra-class compactness of learned representations, and **Mixup Augmentation** to improve generalization across the real-to-synthetic domain gap. This model required fine-tuning on the available training data, using advanced optimization techniques including mixed precision training and cosine annealing scheduling.

#### FaceNet with Face-Specific Preprocessing

Recognizing that the competition domain specifically involved facial images, we implemented **FaceNet (InceptionResnetV1)** pre-trained on VGGFace2 on the day of the competition. This approach was specifically designed after understanding the face-centric nature of the dataset. The implementation included **MTCNN Face Detection** for precise face localization and cropping, **multi-scale data augmentation** incorporating rotation, brightness adjustment, and horizontal flipping, and **identity-aware retrieval** leveraging training data to improve ranking through identity similarity bonuses. Importantly, while ResNet-50 and CLIP models required fine-tuning on the training data, FaceNet leveraged its pre-existing face-specific knowledge without additional fine-tuning, relying instead on enhanced data augmentation and specialized retrieval strategies.

---

## Repository Structure

This repository is organized as follows:

* **`models/`**: This folder contains the trained machine learning models used in the project.
* **`submission/`**: Here you'll find the files related to the final competition submissions.
* **`images_competition/`**: Contains the images used for the main competition.
* **`images_fish/`**: Contains images from a dataset on fish species.
* Other folders (e.g., `images_clothing`, `images_animals`) containing data from preliminary tests are structured similarly, providing a complete reference to the different test domains.

---

## Summary of Results

On the final celebrity face dataset, our approaches achieved the following performance scores:

| Setting                             | Score (out of 1000) |
| :---------------------------------- | :------------------ |
| ResNet-50                           | 75                  |
| CLIP with advanced training         | 880                 |
| FaceNet with face-specific processing | 970                 |
| Ensemble (FaceNet + CLIP)           | 980                 |

The superior performance of **FaceNet** demonstrated the importance of domain-specific pre-training for face recognition tasks. The combination of FaceNet and CLIP in our final ensemble approach achieved near-optimal performance, highlighting the complementary strengths of face-specific and general visual understanding models. The significant improvement from ResNet-50 to the specialized models underscores the critical importance of choosing appropriate architectures for domain-specific retrieval tasks.

---

## Contact

For any questions or clarifications, please feel free to reach out.

**Author:** Elisa Negrini (elisa.negrini@studenti.unitn.it), Michele Lovato Menin (michele.lovato-1@studenti.unitn.it), Tommaso Ballarini (tommaso.ballarini-1@studenti.unitn.it)














RESULTS:
- modello vit_ft.py           ----- t8 ---- 82.22% -- k = 10 -- epochs = 10
- modello vit_ft.py           ----- t8 ---- 81.46% -- k = 10 -- epochs = 5 -- 1min30/epoch
- modello vit.py              ----- t8 ---- 81.30% -- k = 10
- modello clip_ft.py          ----- t8 ---- 81.10% -- k = 10 -- epochs = 10-- 1min/epoch
- modello efficient_ft.py     ----- t8 ---- 80.59% -- k = 10 -- epochs = 5 -- 50sec/epoch
- modello clip_ft.py          ----- t8 ---- 79.75% -- k = 10 -- epochs = 5
- modello dino.py             ----- t8 ---- 79.63% -- k = 10 
- modello dino_ft.py          ----- t8 ---- 76.78% -- k = 10 -- epochs = 5
- modello resnet_ft.py        ----- t8 ---- 76.09% -- k = 10 -- epochs = 5 -- 50sec/epoch
- modello clip.py             ----- t8 ---- 57.49% -- k = 10 
- modello resnet.py           ----- t8 ---- 57.30% -- k = 10 

- modello_dino.py             ----- t7 ---- 95.92% -- k = 50
- modello_minestrone_ CDER.py ----- t7 ---- 94.48% -- k = 50
- model_vit.py                ----- t7 ---- 92.96% -- k = 50
- model_efficient_net_v2_l.py ----- t7 ---- 91.73% -- k = 30
- model_efficient_net_v2_l.py ----- t7 ---- 90.48% -- k = 50
- model_clip_vit_base_..ipynb ----- t7 ---- 79.51% -- k = 49
- model_resnet50.py           ----- t7 ---- 74.96% -- k = 50
- model_convnext.py           ----- t7 ---- 84.96% -- k = 50
- MMM(tutti)                  ----- t7 ---- 96.80% -- k = 50
- MMM(no_Re)                  ----- t7 ---- 96.96% -- k = 50 ~ 8 min
- MMM(no_ReCl)                ----- t7 ---- 96.88% -- k = 50 ~ 7 min
- MMM(no_ReClCo)              ----- t7 ---- 96.96% -- k = 50 
- MMM(no_ReClCoEf)            ----- t7 ---- 96.88% -- k = 50 ~ 3 min
- MMMR(tutti)                 ----- t7 ---- 96.96% -- k = 50 ~ 

- model_convnext.py           ----- t6 ---- 50.42% -- k = 50
- model_efficient_net_v2_l.py ----- t6 ---- 54.42% -- k = 50
- modello_minestrone_ CDER.py ----- t6 ---- 48.42% -- k = 50
- model_clip_vit_base_..ipynb ----- t6 ---- 56.71% -- k = 49
- modello_dino.py             ----- t6 ---- 43.16% -- k = 50
- model_efficient_net_v2_l.py ----- t6 ---- 59.12% -- k = 30
- model_clip_vit_base_..ipynb ----- t6 ---- 66.75% -- k = 20
- model_vit.py                ----- t6 ---- 62.32% -- k = 50
- model_RESNET.py             ----- t6 ---- 54.32% -- k = 50
- MMM(tutti)                  ----- t6 ---- 59.58% -- k = 50 ~ 10 min
- MMM(no_DiCo)                ----- t6 ---- 62.74% -- k = 50 ~ 6 min
- MMM (no_DiCoRe)             ----- t6 ---- 63.47% -- k = 50 ~ 5 min
- MMM (no_DiCoReEf)           ----- t6 ---- 62.42% -- k = 50 ~ 3 min
- MMMR(tutti)                 ----- t6 ----        -- k = 50 

- model_efficient_net_v2_l.py ----- t4 ---- 68.33% -- k = 30 
- modello_dino.py             ----- t4 ---- 61.11% -- k = 30
- modello_vit_ft.py           ----- t4 ---- 42.33% -- k = 50
- model_efficient_net_v2_l.py ----- t4 ---- 42.00% -- k = 50 
- modello_dino.py             ----- t4 ---- 39.33% -- k = 50

- modello_dino.py             ----- t1 ---- 98.33% -- k = 30
- modello_dino.py             ----- t1 ---- 97.75% -- k = 50
- model_resnet50.py           ----- t1 ---- 84.58% -- k = 30
- model_clip_vit_base_..ipynb ----- t1 ---- 82.14% -- k = 49

MMM= modello minestrone mean