
from FlappyBird import flappy as fp
from FlappyBird.flappy import PlayGame, TrainGame, EvaluateGame
from dqn import Agent, train, evaluate
import torch
import os 
import sys

file_dir = os.path.dirname(os.path.realpath(__file__))

supported_types = ['PLAY', 'TRAIN', 'TEST', 'EVALUATE']

def correct_punctuation(supported_Types):
   '''Adds the correct punctuation to the supported types'''
   if len(supported_Types)>2:
      string = ", ".join(supported_Types)
   else:
      string = " ".join(supported_Types)
   string = string.lower()
   string = string.title()
   index_of_last_word = len(string)-len(supported_Types[-1])
   string = string[:index_of_last_word] + 'and ' + string[index_of_last_word:]
   return string

def main(type_game):
   '''Prompts user for what game then runs it'''

   try:
      type_game = sys.argv[1]
   except:
      type_game = None
   
   try:
      fps = int(sys.argv[2])
   except:
      fps = 30
   
   try:
      if sys.argv[3].upper() == 'INFINITE':
         epoch = None
      else:
         epoch = int(sys.argv[3])
   except:
      if type_game != None and type_game.upper() == 'EVALUATE':
         epoch = None
      else:
         epoch = 2500


   if type_game == None:
      type_game = input('Type of Game? ')
      while True:  
         if type_game.upper() not in supported_types:
            print('Only types of games supported are: ' + correct_punctuation(supported_types))
            type_game = input('Type of Game? ')
         else:
            break
   

   if type_game.upper() == 'TRAIN':
      gm = fp.GameManger(TrainGame(file_dir = file_dir), fps_count = 3)
      agent = Agent(gm)
      train(agent, epochs = epoch, plotting_scores=False)

   elif type_game.upper() == 'PLAY':
      gm = fp.GameManger(PlayGame(file_dir = file_dir))
      gm.play()

   elif type_game.upper() == 'EVALUATE':
      gm = fp.GameManger(EvaluateGame(file_dir = file_dir, fps = fps))
      agent = Agent(gm)
      agent.model.load_state_dict(torch.load(file_dir + '/test_weights.pt'))
      evaluate(agent, epochs = epoch)

   elif type_game.upper() == 'TEST':

      gm = fp.GameManger(TrainGame(file_dir = file_dir))
      agent = Agent(gm)
      train(agent,epochs=epoch) #epochs here are the number of games run in order to trains
      gm = fp.GameManger(EvaluateGame(file_dir = file_dir))
      agent.new_game_manager(gm)
      print('starting to evaluate')
      for i in range(3): #number of times it is going to be evaluated
         evaluate(agent, run_num = i, epochs = 250) #epochs here are the number of games used when evaluating 
         


if __name__ == "__main__":
   main('train')

