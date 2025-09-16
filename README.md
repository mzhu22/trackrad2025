# trackrad2025

Team YouBetcha's submission to the [TrackRAD2025 Grand Challenge](https://trackrad2025.grand-challenge.org/).

TrackRAD2025 was an open competition to develop methods for real-time tumor tracking in 2D magnetic resonance imaging (MRI) videos.

Our approach uses [Segment Anything Model 2](https://github.com/facebookresearch/sam2) (SAM2), a foundation model for video object segmentation, and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), an auto-configuring U-Net framework. We also developed a web application for image annotation to generate training data to finetuning SAM2.

Directory structure:

-   `/writeups`: An introductory presentation and a full manuscript detailing the methods
-   `/trackrad-model`: Models used for the inference loop, along with notebooks and scripts for training + eval
-   `/labeling-app`: Web application for semi-automated data annotation

See additional README files within each directory for more information.
