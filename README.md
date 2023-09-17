# LiverSegments-MedSeg
Use 2D U-Net model to perform segmentation of liver segments

# MedSeg Dataset
MedSeg, H. B. Jenssenand T. Sakinis, “MedSeg Liver segments dataset”. figshare, 26-Jan-2021, doi: 10.6084/m9.figshare.13643252.v2.
https://figshare.com/articles/dataset/MedSeg_Liver_segments_dataset/13643252

# Result (Dice coefficient)
Total average: 0.7162
|Fold|  S1  |  S2  |  S3  |  S4a |  S4b |  S5  |  S6  |  S7  |  S8  |
|----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 1  |0.3883|0.7677|0.8020|0.6446|0.7794|0.7919|0.7889|0.7706|0.7748|
| 2  |0.7165|0.8325|0.7559|0.6395|0.6679|0.7434|0.7181|0.7400|0.7032|
| 3  |0.6633|0.8101|0.8221|0.6798|0.7739|0.7231|0.7156|0.6066|0.6748|
| 4  |0.6942|0.7317|0.4900|0.6595|0.6333|0.7586|0.7337|0.7307|0.7652|
| 5  |0.7646|0.4640|0.7769|0.6131|0.6968|0.7786|0.8070|0.8635|0.7738|