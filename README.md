# AI Learns To Play FlappyBird 

Description
-----------

An implementation of Deep Q Learning (DQN) in a Flappy Bird Environment. A [FlappyBird](https://github.com/sourabhv/FlapPyBird) Python implementation was modified in order to create an interface that an RL agent can use. 

Future Features/Improvements
---------------

(Going from most important to least)

1. Clean up code (better variable names, docstrings, and type annotations)

1. Provide more update ```test_weights.pt```

1. Create a Gym Wrapper for the GameMangager

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

How To Run With Command-Line Arguements
-----------------------------

The program can also take command line arguments:

```bash
python -m path\to\AI_Learns_Flappy.py (type_game) --fps (fps) --epochs(epoch)
```
__type_game:__ Can either enter play, train, evaluate, or test.

- Play: Allows the user to play the original implementation
- Train: Trains the agent for the given amount of epochs (episodes) and saves the weights of the model with the best results
- Evaluate: Will load the weights ```test_weights.pt``` found in the current directory for evaluation
- Test: Will train and then evaluate the learned policy

__fps:__ Will be the fps that the game runs at. The time of the game is tied to the fps so increasing the fps will increase the speed the game runs at.
If not provided the program will default to 30 fps.

__epoch:__ Will be the amount of epochs the agent will either train, evaluate, or train during test.
If not provided the program will defalut to 2500 epochs.
If the word ```Infinite``` is put in, there will be no limit to the epochs





Orginal Repositiories
-------------

Orginal repositiories used in order to make this project

- [FlapPyBird](https://github.com/sourabhv/FlapPyBird) 
