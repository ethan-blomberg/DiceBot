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

<p align="center">
  <blockquote class="imgur-embed-pub" lang="en" data-id="a/CCzMNnw"  ><a href="//imgur.com/a/CCzMNnw">Dice results</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
</p>

## License

[MIT](https://choosealicense.com/licenses/mit/)
