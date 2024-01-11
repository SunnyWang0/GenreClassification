# Music Genre Classification using CNNs with Dual-Spectrogram Data Augmentation

## Overview

This project introduces an innovative approach to music genre classification by leveraging Convolutional Neural Networks (CNNs). Our method uniquely combines Melspectrograms and Short-Time Fourier Transform (STFT) spectrograms as joint data inputs. This approach aims to significantly enhance classification accuracy in music genre identification. 

See the paper for a more detailed analysis of the model.

## Key Features

- **Dual-Spectrogram Input:** Utilizes both Melspectrograms and STFT spectrograms, offering a comprehensive audio analysis.
- **DenseNet Architecture:** Compares DenseNet in both pre-trained and non-pre-trained configurations, highlighting the benefits of transfer learning and fine-tuning.
- **Transfer Learning with DenseNet121:** Demonstrates substantial improvements in classification accuracy when using the pre-trained DenseNet121 model.

## Results

Our experiments have shown notable enhancements in music genre classification, evidenced by:
- Improved test accuracy.
- Higher Area Under the Receiver Operating Characteristic (AUROC) scores.
- More accurate confusion matrix results, particularly noticeable with transfer learning using the pre-trained DenseNet121 model.

## Conclusion

The dual-spectrogram augmentation method presented here opens new avenues for music genre classification. By effectively combining different spectrogram types and leveraging the power of transfer learning, our approach sets a new benchmark for accuracy in this field.
