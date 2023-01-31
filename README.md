# DiceBot
Dice rolling machine using computer vision to measure statistical fairness.

I’m doing everything in python on a Jetson Nano, with a pi cam v2.	

The mechanical portion of the dice machine is tested and working. The actuation (belt lifter and tray tilter) will be offloaded to a microcontroller via serial communication. Otherwise I’d have to look into multithreading and risk losing computation power.

Watch this quick [video](https://photos.app.goo.gl/aGJm1UFAFVEBBskr6) of the machine in action

## Goal
I have an end goal of recording results of any die roll - D20 with digits, cubes with pips, etc. - but immediate goal of recognizing this specific D20 die:

<p align="center">
  <img src="https://static.wixstatic.com/media/32aaac_27d38b84839e4b10b9d41b67a02a75b9~mv2.jpg/v1/fill/w_500,h_500,al_c,q_85,usm_0.66_1.00_0.01/32aaac_27d38b84839e4b10b9d41b67a02a75b9~mv2.webp" width = "250" height = "250"/>
</p>
<p align="center">
Brass D20 from precisionplaydice.com
</p>

## Big Questions
* Detect dice face (1-20)  - 20 outputs, or individual digits (0-9) - 10 outputs ?
* Can both be generalized to categorize dice with pips?
* If I only have 10 outputs, how do I then move to multi-digit numbers? 
  
  > I believe I’d need a preprocess to identify the number of digits, extract each, give each to CNN, then recombine to create a two-digit number. It sounds more complex than just designing the model to have 20 outputs.

## Where I Am Now

### Preprocessing
Capture image -> grayscale -> blur -> edge detection -> circle detection -> crop 

> Note: Circle detection will most-likely only work on D20s. Need to find a general method for all dice

Now I have a small ~200px gray image of just the die, and a copy of the original image with the die outlined.

Optionally, repeat circle detection and crop on the small image, to get ~40px image of just the digit(s). This currently works for my specific D20. I am confident an MNIST-trained model could recognize faces 1-9. All my parameters for the various image processing methods seem to work fine but may be optimized. I’m not sure how to do this besides trial and error (which is how I found my current params).

> This preprocess will be used for creating a labeled data set for training, or transfer learning later. I can roll dice anywhere in the camera’s FOV and it will return roughly the same size image of just the die. Pretty satisfying.

### Creating Datasets for Training
Start with the first face and capture an image, preprocess (above) and resize to ~32px and save to a directory named with the label. Repeat while adjusting lighting and rotation with external peripherals, as well as digitally skewing and rotating each of these in post-processing to create a larger data set. After ~1000 images are saved in the labeled directory, prompt the user to rotate to the next face, then repeat. 

> This code is complete minus post-processing but has not been executed. 

### Preparing Datasets for Training
Extract a random ~20% of images from each directory to set aside for testing/validation. 

Flatten image matrix and label at the 0-index. Should be of length equal to the number of pixels + 1 for its label. 

Add this vector to the main training data matrix to have dimensions: number of pixels+1 by number of images.

> This code is complete but has not been executed. 

### Neural Network
Currently working from scratch with just numpy. I found a good tutorial for this using MNIST as it’s example. I may switch to Tensorflow or similar for time/complexity constraints. 

## License

[MIT](https://choosealicense.com/licenses/mit/)
