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



## License

[MIT](https://choosealicense.com/licenses/mit/)
