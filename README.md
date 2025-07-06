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

The second approach employed **CLIP (ViT-L-14)** with sophisticated training enhancements. We developed an initial version of this model before the competition, but refined its architecture afterwards. The current custom architecture, named CLIPArcFace, combines the powerful Vision Transformer backbone with ArcFace Loss to learn highly discriminative embeddings. At its core, the model leverages the ViT-L-14 from OpenCLIP as its backbone. To efficiently train this large model on our specific task, only the
last few transformer layers of the CLIP visual encoder are unfrozen and fine-tuned. Following the backbone, a projection head transforms the features into a refined 1024-dimensional embedding space. Finally, to enhance the discriminative power of the learned embeddings, ArcFace Loss is utilized.

#### FaceNet with Face-Specific Preprocessing

Recognizing that the competition domain specifically involved facial images, we implemented **FaceNet** (Inception-ResnetV1 pre-trained on VGGFace2) on the day of the competition. This approach was specifically designed after understanding the face-centric nature of the dataset. The implementation included MTCNN Face Detection for precise face localization and cropping, multi-scale data augmentation incorporating rotation, brightness adjustment, and horizontal flipping. Importantly, while ResNet-50 and CLIP
models required fine-tuning on the training data, FaceNet leveraged its pre-existing face-specific knowledge without additional fine-tuning, relying instead on enhanced data augmentation and specialized retrieval strategies.

---

## Repository Structure

This repository is organized as follows:

* **`models/`**: This folder contains the trained machine learning models used in the project.
* **`images_competition/`**: Contains the images used for the main competition.
* **`images_fish/`**: Contains images from a dataset on fish species.
* Other folders (e.g., `images_clothing`, `images_animals`) containing data from preliminary tests are structured similarly, providing a complete reference to the different test domains.

---

## Summary of Results

On the final celebrity face dataset, our approaches achieved the following performance scores:

| Setting                             | Score (out of 1000) |
| :---------------------------------- | :------------------ |
| ResNet-50                           | 74.29               |
| CLIP with advanced training         | 910.17              |
| FaceNet with face-specific processing | 970.45              |
| Ensemble (FaceNet + CLIP)           | 973.95              |

The superior performance of **FaceNet** demonstrated the importance of domain-specific pre-training for face recognition tasks. The combination of FaceNet and CLIP in our final ensemble approach achieved near-optimal performance, highlighting the complementary strengths of face-specific and general visual understanding models. The significant improvement from ResNet-50 to the specialized models underscores the critical importance of choosing appropriate architectures for domain-specific retrieval tasks.

---

## Contact

For any questions or clarifications, please feel free to reach out.

**Author:** Elisa Negrini (elisa.negrini@studenti.unitn.it), Michele Lovato Menin (michele.lovato-1@studenti.unitn.it), Tommaso Ballarini (tommaso.ballarini-1@studenti.unitn.it)
