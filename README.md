# AI Learns To Play FlappyBird 

Description
-----------

This project uses Deep Q learning in order to teach an AI how to play Flappy Bird. In order to do this a FlappyBird [clone](https://github.com/sourabhv/FlapPyBird) as well as a Deep Q learning [model](https://github.com/python-engineer/snake-ai-pytorch) was used. Code from both of these projects had to be changed. The game had to be reorganized so that each frame could individually be called. A gamemanager was also added to the game which would get the current state of the game, pass in actions from the AI, return the rewards from these actions, run the game, as well as perform some other helpful functions. This was done so that the AI could not only interact with the game but also run the game. This allowed the AI and the game to run in the same thread rather than in different ones, making the program less complex. The model was changed slightly in order to have a larger buffer size and remove the short term memory training. The larger buffer size uses more memory but will also increase performance of the model. Removing the short term memory training caused there to be less instability (an example of instability is the AI learning to only fly down or up). The AI was also made to run the game at 960 fps rather than the native 30 fps in order to train faster.

Future Features
---------------

(Going from most important to least)

1. Increasing the stability of the AI 

1. Add an evaluate mode 

1. Add graphs to help better understand the training process

1. Include pretrained weights so user does not have to train in order to see an evaluation

Current Scores
--------------

Record: 297
Best Average Score: 46.152

Setup (Tested on Windows with Python 3.10.5)
--------------------------------------------

```bash
git clone https://github.com/BananaBob-IQ/Flappy_Bird_AI
```

or download the zip and extract

Then

```bash
cd Flappy_Bird_AI
pip install -r requirements.txt
```

How To Use
----------

In the ```\path\to\Flappy_Bird_AI``` directory run:

```bash
python -m AI_Learns_Flappy.py
```

When prompted in the terminal type in either:

```bash 
Type of Game? play
```

to play the game using <kbd>&uarr;</kbd> or <kbd>Space</kbd> keys to play

or

```bash
Type of Game? train
```

to train the AI 

When training, the AI will save its best model in ```\path\to\Flappy_Bird_AI\model```
In this folder there will be a pt file of the model as well as a txt with its best score that it achieved

When the program is launched you may need to change window focus to newly made window

In order to exit the program use the <kbd>Esc</kbd> key

Orginal Repositiories
-------------

Orginal repositiories used in order to make this project

- [FlapPyBird](https://github.com/sourabhv/FlapPyBird)
- [snake-ai-pytorch](https://github.com/python-engineer/snake-ai-pytorch)