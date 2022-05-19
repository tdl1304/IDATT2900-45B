# Damage script

``python main.py --batch_size=1 --mode=0 --max_dmg_freq=-1 --load_model_path="models/model_999500"``

## Types of damage

* Gaussian blur
Requires sigma value. Greater sigma equals more blurryness
default argument: --sigma = 0.2
mode 0

* Removal of pixels
mode 1

* Adversarial attacks
Requires epsilon value. Greater epsiolon equals more noise.
default argument: --eps=0.007
mode 2

## Parameters:  
optional arguments:  
  -h, --help            show this help message and exit  
  --mode MODE           0 for gaussian blur, 1 for pixel removal, 2 for adversarial attacks  
  --n_iterations N_ITERATIONS Number of iterations to test for.  
  --batch_size BATCH_SIZE  
  Batch size.  
  --n_channels N_CHANNELS  
                        Number of channels of the input tensor  
  --dmg_freq DMG_FREQ   Frequency for damaging  
  --max_dmg_freq MAX_DMG_FREQ  
                        Limit the number of times of damaging, -1 if not specified       
  --sigma SIGMA         Sigma for gaussian blur, greater value means more blurry  
  --alpha 0.005         Alpha for how much noise to add  
  --padding PADDING     Padding for image  
  --eps EPS             Epsion scales the amount of damage done from adversarial attacks  
  --img 🐰               The emoji to train on  
  --size SIZE           size of image  
  --logdir LOGDIR       Logging folder for new model  
  --load_model_path LOAD_MODEL_PATH  
                        Path to pre trained model
