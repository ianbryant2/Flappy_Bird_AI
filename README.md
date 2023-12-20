# AI Learns To Play FlappyBird 

Description
-----------

This project uses Deep Q learning in order to teach an AI how to play Flappy Bird. In order to do this a FlappyBird [clone](https://github.com/sourabhv/FlapPyBird) as well as a Deep Q learning [model](https://github.com/python-engineer/snake-ai-pytorch) was used. Code from both of these projects had to be changed. The game had to be reorganized so that each frame could individually be called. A gamemanager was also added to the game which would get the current state of the game, pass in actions from the AI, return the rewards from these actions, run the game, as well as perform some other helpful functions. This was done so that the AI could not only interact with the game but also run the game. This allowed the AI and the game to run in the same thread rather than in different ones, making the program less complex. The model was changed slightly in order to have a larger buffer size and remove the short term memory training(training the agent after each step). The larger buffer size uses more memory but will increase the performance of the model. Removing the short term memory training caused there to be less instability (an example of instability is the AI learning to only fly down or up). The AI was also made to run the game at 960 fps rather than the native 30 fps in order to train faster.

Future Features/Improvements
---------------

(Going from most important to least)

1. Seperate view from model in the game in order to train faster

1. Able to load weights from previous training 

1. Improve performance of trained AI

1. Increase stability during training

Current Scores
--------------

Record: 635

Best Average Score: 56.312

Setup (Tested on Windows with Python 3.10.11)
--------------------------------------------

(Note there may be issues installing the dependencies on later versions of Python)

```bash
git clone https://github.com/ianbryant2/Flappy_Bird_AI
```

or download the zip and extract

Then

```bash
cd Flappy_Bird_AI
pip install -r requirements.txt
```

How To Use
----------

Run:

```bash
python -m path\to\main.py
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

When training, the agent will save its best model in ```\path\to\Flappy_Bird_AI\model```
In this folder there will be a pt file of the model as well as a txt with its best score that it achieved

or

```bash
Type of Game? evaluate
```

to evaluate the included pretrained weights

When evaluating information will be presented. The numbers in the bottom left corner represents the output of the program where if the top number is larger the agent will wait and if the bottom number is larger the agent will flap. The agent will save its information about its scores to ```\path\to\Flappy_Bird_AI\model```

When the program is launched you may need to change window focus to the newly made window

In order to exit the program use the <kbd>Esc</kbd> key

How To Run With Command-Line Arguements
-----------------------------

The program can also take command line arguments:

```bash
python -m path\to\main.py (type_game) (fps) (epoch)
```
__type_game:__ Can either enter play, train, evaluate, or test. Every type except test has the same functionality as described above. Test will train an AI for a certain amount of epochs and then evaluate it for 250 epochs.
If not provided the program will prompt the user for the game.

__fps:__ Will be the fps that the game runs at. The time of the game is tied to the fps so increasing the fps will increase the speed the game runs at.
If not provided the program will default to 30 fps.

__epoch:__ Will be the amount of epochs the agent will either train, evaluate, or train during test.
If not provided the program will defalut to 2500 epochs.
If the word ```Infinite``` is put in, there will be no limit to the epochs





Orginal Repositiories
-------------

Orginal repositiories used in order to make this project

- [FlapPyBird](https://github.com/sourabhv/FlapPyBird)
- [snake-ai-pytorch](https://github.com/python-engineer/snake-ai-pytorch)
