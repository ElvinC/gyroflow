---
layout: single
title: "Gyroflow guide"
---

### In-depth video guide:

<iframe width="560" height="315" src="https://www.youtube.com/embed/f4YD5pGmnxM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### Run using python:

* Install [poetry](https://python-poetry.org/docs/#installation)
* Clone or download the files from the [Gyroflow repository](https://github.com/ElvinC/gyroflow)
* Navigate to the folder using a commandline and install dependencies using `poetry install`
* Run the application using `poetry run python gyroflow.py`


[Youtube channel with updates](https://www.youtube.com/channel/UCr0Hh-AUc4CU-36yTjx-lTA)

[FPV stabilization tools FB group](https://www.facebook.com/groups/fpvtools)

[RSgoBlackbox discord server](https://discord.gg/2He3XTjtpt)

### Motion data sources
* GoPro metadata
* Insta360 metadata
* Main drone flight controller
* Auxilliary drone flight controller
* Runcam 5 Orange or iFlight GOCam GR
* Custom logger electronics.

### Video sources
* Low rolling shutter is preferred (less jello)
* Highest possible field of view for more room to play with. This means 4:3 aspect for 4:3 sensors.
* High-ish shutter speed decreases stabilization-induced motion blur. E.g. 90 degree shutter is usually fine. 180 degree shutter can give decent results if the source was mostly smooth.
* Few/no dropped or duplicated frames in the footage. Can be fixed through processing in some cases.
* Square or linearly stretched pixels. Anamorphic footage is untested.