# TrackRAD2025 (Mayo Clinic Radiation Oncology)

TrackRAD2025 was an open competition to develop methods for real-time tumor tracking in 2D magnetic resonance imaging (MRI) videos.

Our approach uses [Segment Anything Model 2](https://github.com/facebookresearch/sam2) (SAM2), a foundation model for video object segmentation. We also developed a web application for image annotation to generate training data for finetuning SAM2.

Directory structure:

-   `/trackrad-model`: Models used for the inference loop, along with notebooks and scripts for training + eval
-   `/labeling-app`: Web application for semi-automated data annotation
-   `/notebooks`: Statistical analysis of model performance

See additional README files within each directory for more information.
