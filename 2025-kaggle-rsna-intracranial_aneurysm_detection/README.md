# Approach Overview and Key Learnings

**RSNA - Intracranial Aneurysm Detection**



**Data Curation**

1. **Incorrect DICOM rescaling**: Some files have erroneous slope/intercept values. For CT scans, the raw pixel values (before rescaling) may already represent true HU values. For MRI, this isn't critical since pixel values lack standardized meaning - either applying standard conversions or min-max normalization is acceptable. **Solution from Ian**: May not resolve all cases, but visual inspection of slices containing targets shows no major issues.

   ```python
   def should_rescale_ct(ds, pixel_array):
       """Determine if CT should be rescaled"""
       if ds.get('Modality', '') != 'CT':
           return False
       if not (hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept')):
           return False
       min_pixel = pixel_array.min()
       if min_pixel >= -100 or min_pixel == -2000:
           return True
       return False
   ```

2. **Scout/projection images**: Some series contain scout views or maximum intensity projections. These were removed in Data Update 2 by excluding images with inconsistent row/column dimensions compared to the main series.

3. **Missing series folders**: Some cases in train_localizer.csv have targets but no corresponding series folder

   ```python
   "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444"
   ```

4. **Missing localizations**: some positive cases in train.csv lack localizations in train_localizer.csv.

   ```python
   "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068"
   "1.2.826.0.1.3680043.8.498.12937082136541515013380696257898978214",
   "1.2.826.0.1.3680043.8.498.86840850085811129970747331978337342341",
   ```

5. **Duplicate coordinates**: Same series ID and coordinates appear with different locations, making target location ambiguous.

   ```python
   "1.2.826.0.1.3680043.8.498.10733938921373716882398209756836684843"
   ```

6. **Data update inconsistencies**: Two series changed in Data Update 2 but reverted - unclear if issues remain.

   ```python
   "1.2.826.0.1.3680043.8.498.11292203154407642658894712229998766945"
   "1.2.826.0.1.3680043.8.498.74390569791112039529514861261033590424",
   ```

7. **Labeling errors**: some series have visually incorrect labels in localizer.csv.

   ```python
   "1.2.826.0.1.3680043.8.498.99892390884723813599532075083872271516",
   "1.2.826.0.1.3680043.8.498.99421822954919332641371697175982753182",
   "1.2.826.0.1.3680043.8.498.93005379507993862369794871518209403819",
   "1.2.826.0.1.3680043.8.498.87133443408651185245864983172506753347",
   "1.2.826.0.1.3680043.8.498.85042275841446604538710616923989532822",
   "1.2.826.0.1.3680043.8.498.75294325392457179365040684378207706807",
   "1.2.826.0.1.3680043.8.498.73348230187682293339845869829853553626",
   "1.2.826.0.1.3680043.8.498.34908224715351895924870591631151425521",
   "1.2.826.0.1.3680043.8.498.13299935636593758131187104226860563078",
   "1.2.826.0.1.3680043.8.498.12780687841924878965940656634052376723",
   "1.2.826.0.1.3680043.8.498.12285352638636973719542944532929535087",
   "1.2.826.0.1.3680043.8.498.10820472882684587647235099308830427864",
   "1.2.826.0.1.3680043.8.498.11019101980573889157112037207769236902",
   "1.2.826.0.1.3680043.8.498.13356606276376861530476731358572238037",
   "1.2.826.0.1.3680043.8.498.81867770017494605078034950552739870155"
   ```

8. **Multi-frame orientation errors**: All multi-frame images default to 'FH-AP-RL' orientation, but many actually have different orientations (AP-HF-RL or RL-HF-AP)

   ```python
   1: 'AP-HF-RL'
   1.2.826.0.1.3680043.8.498.10862138275035843887055171875480735964
   1.2.826.0.1.3680043.8.498.11396958000946738156009956455739305762
   1.2.826.0.1.3680043.8.498.11915319973409844345177713085783065237
   1.2.826.0.1.3680043.8.498.12163038646729971461006564302880090481
   1.2.826.0.1.3680043.8.498.12754621213831983134209152548119057365
   1.2.826.0.1.3680043.8.498.13001629435974764211403087597568806527
   1.2.826.0.1.3680043.8.498.13334658148703615392388818414999249292
   1.2.826.0.1.3680043.8.498.18831402822041226140887003611379903167
   1.2.826.0.1.3680043.8.498.29351212950805314631667854934458469754
   1.2.826.0.1.3680043.8.498.35123157147325830213906326339070528034
   1.2.826.0.1.3680043.8.498.36861937197087749960171145883205456895
   1.2.826.0.1.3680043.8.498.40006562159206402632477316663171307697
   1.2.826.0.1.3680043.8.498.46538678358294253983538640149161986964
   1.2.826.0.1.3680043.8.498.49672398100697832208944634471809461961
   1.2.826.0.1.3680043.8.498.50668879928342593291812487079769153076
   1.2.826.0.1.3680043.8.498.50916621085656781540278427064467759139
   1.2.826.0.1.3680043.8.498.67129993505475797984506180089478722899
   1.2.826.0.1.3680043.8.498.85431182782929944864196573042506906105
   1.2.826.0.1.3680043.8.498.85694228896758469614431673786651945288
   1.2.826.0.1.3680043.8.498.91280907751913581577764343702856084945
   1.2.826.0.1.3680043.8.498.93156694293030030637766074579373694728
   1.2.826.0.1.3680043.8.498.99804081131933373817667779922320327920
   
   2: 'RL-HF-AP'
   1.2.826.0.1.3680043.8.498.11887329867812275491160566603814454129
   1.2.826.0.1.3680043.8.498.21275250875812455389777450891502640750
   1.2.826.0.1.3680043.8.498.27235396640484934153639773593945542938
   1.2.826.0.1.3680043.8.498.35440393683691371542782507480292365786
   1.2.826.0.1.3680043.8.498.56222999331067503423242588210365055932
   ```

9. **Inconsistent Z-spacing**: Slice thickness/pixel spacing sometimes incorrect with abnormally large values or skipped slices (e.g., 5mm→10mm jumps).

10. **Segmentation Data**: First confirmed that all DICOM files corresponding to NII files are not multi-frame. However, issues remain - approximately 170 out of 177 are usable. Problems include dimension mismatches between NII files and their corresponding DICOM series.

    ```python
    # Dimension Mismatch
    "1.2.826.0.1.3680043.8.498.12271269630687930751200307891697907423",
    "1.2.826.0.1.3680043.8.498.14375161350968928494386548917647435597",
    "1.2.826.0.1.3680043.8.498.50369188120242587742908379292729868174",
    "1.2.826.0.1.3680043.8.498.54865110953409154322874363435644372368",
    "1.2.826.0.1.3680043.8.498.68654901185438820364160878605611510817",
    "1.2.826.0.1.3680043.8.498.75294325392457179365040684378207706807",
    "1.2.826.0.1.3680043.8.498.97256479550884529885940791074752719030",
    ```

11. **Annotation errors in segmentation data**: Some remaining cases have incorrectly placed annotations in unexpected locations, causing significant errors when performing ROI cropping. For example, after LPS alignment, slice 102 in series `1.2.826.0.1.3680043.8.498.23047023542526806696555440426928375679` and certain slices in series `1.2.826.0.1.3680043.8.498.11938739392606296532297884225608408867` show misplaced annotations.



**Data Preprocessing**

The dataset contains a mix of CT, MR, angiography data, and bone subtraction images. While pure CT data can be converted to Hounsfield Units (HU) with specific windowing, this multi-modality dataset required a different approach—I opted for min-max normalization.

Initially, I applied min-max normalization across entire volumes, but this created ring artifacts when reconstructing ROIs. Converting to uint8 resulted in low contrast and information loss. The final approach normalized only the cropped regions containing both the Circle of Willis (COW) and targets.



**ROI Cropping**

I limited ROI cropping to the entire COW region (extended to include all targets) rather than cropping individual vessel branches for several reasons:

- Uncertainty about whether 3D crops of different vessels might overlap
- Vessels are relatively small, and some image dimensions have very high resolution, potentially making them difficult to distinguish
- Risk of missing small structures

However, top solutions successfully classified each vessel individually. I only attempted this approach in the final two days and encountered issues with many targets falling outside ROIs (possibly a code bug—1st place reported only ~10 cases dataset-wide). I abandoned this and continued using the full COW ROI, applying cropping only in the x-y plane while preserving the entire z-axis. I noticed that some volumes extended far in the z-direction (sometimes reaching the lungs), causing ROI model misclassifications. In retrospect, worrying about these edge cases was overthinking. According to [RihanPiggy's solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/11th-place-solution), whole-ROI approaches can still achieve good results.

In the end, I trained a 3D model using EfficientNetB2 with:

- Global min-max normalization across entire volumes
- Volumes resized to 256×256×256
- 170 masks to determine ROI boundaries
- Manual verification: if targets were just outside the ROI, I expanded it to include them (distant targets were excluded)
- Center point and three-directional lengths calculated to derive start/end indices for appropriate cropping during inference

The dataset contained ambiguous 3D input orientations—only three orientations (axial, coronal, sagittal) rather than all eight possible combinations. I tried augmenting training data by converting each orientation to the other two, but this didn't significantly improve cropping accuracy. To ensure adequate ROI coverage, I added a volume coverage ratio loss function to the coordinate MSE loss.



**Classification Model**

I used a 2D model with 3-channel input and CoaT backbone for classification:

- Input: each plane cropped and resized to 224×224, with full z-axis preserved
- Simple training logic: AdamW optimizer (lr=1e-4, weight decay=1e-6), BCE loss for classification
- To ensure clean training data: positive series only trained on slabs containing targets; negative series randomly sampled
- Each slab (3 slices) generated an attention map by convolving a Gaussian kernel (std=5) at each target coordinate, then max-aggregating across all targets (tried std values of 3, 8, 10—minimal difference)
- Added segmentation task alongside classification; training loss decreased after 2-3 epochs
- Loss function: classification BCE + 0.7×segmentation BCE + 0.3×Dice loss
- Final prediction: moved slabs at different steps, predicted separately, then took maximum aggregation

I attempted using RNNs to combine multiple slab features before prediction, but re-predicting all volumes for training took ~40 minutes per epoch. I was unsure about re-training on the validation set and didn't pursue this extensively due to time constraints—it showed only marginal improvement.

I also tried adding an auxiliary vessel segmentation task using pre-segmented vessels from nnU-Net, but found no improvement and felt it over-complicated the multi-task setup.



**Vessel Segmentation**

I trained an nnU-Net but didn't deploy it for inference:

- Input: volumes with min-max normalization resized to 256×256×256 (same as crop model input, though nnU-Net actually uses 128×128×128 patches, making cube dimensions unnecessary)
- Right/Left Posterior Communicating Arteries were difficult to segment—some folds achieved 0.4-0.5, while others remained at 0.0 even after 1000 epochs
- Disabled nnU-Net's default left-right mirroring augmentation by modifying the Trainer file
- All inputs reoriented to LPS orientation



**Potential Directions for Improvement**

1. **ROI Cropping**: Could be more refined
2. **Training Data Balance**: Compared to the [4th place solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/4th-place-solution), my positive case ratio was too high, likely causing overfitting. Their pseudo-labeling distillation approach was clever—predicting on unsampled negative cases, identifying high-confidence false positives, then adding them to the training set.





