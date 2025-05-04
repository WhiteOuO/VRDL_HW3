# VRDL_HW3
## Instance_segmentation
Introduction:
This task is an advanced version of the previous one. In addition to detecting object positions and counts, the model now needs to predict precise contours of the objects, so I incorporated the mask branch into the architecture.  
## Base training and prediction procedure:  
1. **Image Cropping**  
   Use `pic_crop` to read the original dataset folder. This script will crop both the original images and the corresponding masks into four quadrants.

2. **Copy-and-Paste Augmentation**  
   Use `copy_and_paste.py` to read the folder containing the cropped images. It will generate 200 new samples and store them, along with the cropped images, into a new folder named `train_mixed`.

3. **Data Splitting**  
   Run `data_split.py` to divide the nearly 1000 samples in `train_mixed` into 10 smaller subfolders for better training efficiency and potential cross-validation.

4. **Model Training**  
   Start training with `train_aa101.py`. This will generate model checkpoints. You can stop training manually at any time. Afterward, collect all generated models into a single folder for inference.

5. **Prediction**  
   Use `predict101.py` to load all the models from the model folder. This script will generate one prediction JSON file for each model.
## Results  
![image](https://github.com/user-attachments/assets/52db6024-661e-4cec-9c26-91842c188c66)
## Training and Prediction for Each Cell Type

1. **Prepare Per-Class Datasets**  
   First, run `cell_train_split.py` to generate four folders, each containing training data for one specific cell type.

2. **Train One Class at a Time**  
   Use `train_aa101d_class.py` to train a model for one of the specific cell types.  
   By default, it is set to train for **class 1**.  
   You can change the training class by modifying the related settings as shown below:

   ![Training Class Setting](https://github.com/user-attachments/assets/e0cc2d0f-a09e-42fe-b1d6-3bea79ade1a0)  
   ![Training Class Option](https://github.com/user-attachments/assets/c1bd92af-9e63-4244-97f7-152cce271a72)

3. **Modify Prediction Script**  
   To perform prediction, use the existing `predict101.py` script, but make the following change:  
   Set `num_classes = 2` in the model constructor:

   ![Set num_classes = 2](https://github.com/user-attachments/assets/890da1d7-fdfc-438e-990a-65fef41aa38f)

4. **Run Predictions for Each Model**  
   You can choose any trained models from different classes and run predictions with them individually.  
   Each class-specific model will output predictions with label **1** (this is expected).  
   Afterward, manually replace label `1` in the output JSON files with label `2`, `3`, or `4`, depending on which class the model is trained to predict.

5. **Merge Predictions**  
   Once you have four JSON prediction files—one for each class—run `prediction_combine.py` to merge them into a single output.


