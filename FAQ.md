---
layout: single
title: FAQ
---

## Something doesn't work and I didn't watch the tutorial video in full. Halp!?
Apparently only 15% of viewers watched the full video. Please do so if you haven't before asking for help since it contains a lot of tips and tricks: [tutorial](https://youtu.be/f4YD5pGmnxM). 

## My export doesn't work with hardware acceleration?
Make sure you have FFmpeg installed. If problem persist, export with debug info enabled, which should narrow down the problem.

## I'm getting an error with `Unable to read multiple frames` when trying to render a video
Sometimes there are a couple frames right at the beginning of video files which can't be read for some reason. Try setting the render range to a different part of the video from where the error occured.

## I'm processing footage from the Runcam 5 Orange or iFlight GOCam with native gyro logging. Any tips?
Yes! [Have a look here](https://docs.google.com/document/d/1mLrMd8itLYiGNdQh4oWqz-s3g16bfu-TkQNCS_ZqOnA/edit#).

## Which smoothing method should I use? Or: The render is all zoomed in!?!?
Default, 3D smoothing: For general use, but doesn't work for fast moves (image movement is not limited)

Yaw pitch roll smoothing: Lets you decide how much you're smoothing in each axis. It can look nicer to smooth more in the roll axis.

Horizon lock: Attempts to keep the horizon level. Works best for flight without fast rotational moves since some gyros can be maxed out.

3D smoothing with smooth angle limit (Aphobius): Limits how much stabilization is applied during fast flips/rolls. The best option for "freestyle" FPV flight.

3D smoothing with sharp angle limit: Similar to the above, but a different (worse) implementation. Use 3D smoothing with smooth angle limit instead.

Nothing: Does nothing

## Trying to stabilize footage from my camera with IBIS results in footage that still wobbly sometimes.
Sorry mate, the sensor is bouncing around so the gyro doesn't match the image anymore. Nothing can be done on the software side, but if you're feeling adventurous, there are supposedly options to lock the sensor in place with glue or 3D prints.

## I want to try this with Sony metadata _or_ Gyroflow is unable to extract gyro data from GoPro files.
Use this program: https://github.com/AdrianEddy/telemetry-parser by Adrian Eddy to convert the metadata to the blackbox format, and use that in Gyroflow.


## Trying to export gives `FileNotFoundError: [Errno 2] No such file or directory: '/xxx/tttt.mp4_a.mp4' -> '/xxx/tttt.mp4'`
Can happen with some setups, try disabling audio export as a temporary fix.


## I get a weird JSON decode error during processing
You probably accidentally downloaded the whole HTML website instead of the json preset. I recommend downloading and placing the camera_presets folder from the downloads page, and place it together with the executable.

## Will this work with a GoPro Hero 5/7 on a drone
Hero 7 should work fine with Gyroflow for handheld footage, but has fundamental (hardware?) issues with gyro noise and aliasing, so motion data is essentially lost when hardmounted on a drone due to the vibrations. You can use softmounting e.g: [https://www.thingiverse.com/thing:3842261](https://www.thingiverse.com/thing:3842261) or use an external gyro source.

Something similar is the case for the SMO4k, which also has noisy motion data.


## Is this better than Reelsteady Go for GoPro?
No.

## Is this better than SteadXP?
Probably not, but I haven't tried SteadXP.

## Any other tools/ressources I should know about?
* [BlackboxToGPMF](https://github.com/Cleric-K/BlackboxToGPMF/tree/gui) by Attilafustos and Cleric-K for emulating GoPro files for use with Reelsteady Go. Works best for action-style cameras with similar distortions as the GoPro.
* [blackbox2gpmf](https://github.com/jaromeyer/blackbox2gpmf) By jaromeyer. Similar to BlackboxToGPMF but specifically for Hero 7 files.
* [Virtual Gimbal](https://github.com/yossato/virtualGimbal) by Yoshiaki Sato. Another gyro-assisted stabilization tool that is in many ways much more advanced than Gyroflow at the moment (has rolling shutter correction). Although it has no graphical UI and I haven't tried it myself.
* [FPV stabilization tools Facebook group](https://www.facebook.com/groups/fpvtools)
* [RSgoBlackbox discord server](https://discord.gg/2He3XTjtpt)
* I also know that at least two other people are working on something similar, so that's pretty neat.

## Are these questions actually "frequently asked"?
No, but they could be.

## How does gyro-based image stabilization work?
Not sure how many are wondering this, but I'm going to answer anyways :). Classic video stabilization methods work by estimating the camera motion using optical flow, but this involves "guesswork" so to speak.
MEMS gyroscopes can be used to determine the rotation of the camera very accurately. The first step is to sync the gyro and the video. Then a virtual camera can be created that moves just like the physical camera, which "projects" the image onto an imaginary plane or sphere. Another (smoothed) camera can recapture the view, resulting in a stabilized video. In practice the projection and recapture can be combined using some fancy mathematics. 

This [companion video](https://youtu.be/I54X4NRuB-Q) to the the paper _Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes_ by Karpenko et al. explains the basic concepts quite well. 

## I can't get X to work, what do?
Watch the [tutorial](https://www.youtube.com/watch?v=f4YD5pGmnxM) if you haven't and try asking in the Discord or Facebook group.

## There is no lens profile for X. What do I do?
Create a new lens profile by following the guide. It only takes a few minutes. If it works well, please submit it to be added as a default profile so others can use it.
Since you can add a name to the profile, future users can tell who the awesome calibrator was.

## I tried out a lens profile for my camera, but it doesn't work very well?
Try making a new lens profile to see if that helps. If it works better, please submit the new profile as well. Bad lens profiles typically move too much or too little during panning or tilting, and has "wobbling" at the edges.

## Does this work for DSLR/handheld footage?
It works *alright* for wide angle footage, but rolling shutter artifacts will be present in handheld/walking footage, especially at longer focal lengths.

## Will rolling shutter correction be added?
Soon™

## Why did you spend so much time/effort setting up a nice website and all that instead of adding features to Gyroflow?
Ultimately I'm working on all of this to try out/learn about various things, whether that be computer vision, image processing, orientation math, or in this case Jekyll. Since I don't actually have a serious need for a fancy stabilization tool, the goal is instead to have "fun" and make cool stuff, not rush out computer applications. I do admit that the website looks decent, but that's all thanks to the [Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/).