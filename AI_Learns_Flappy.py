
from FlappyBird import flappy as fp
from FlappyBird.flappy import PlayGame, TrainGame, EvaluateGame
import flappyModel as fm
import torch
import os 
import sys

file_dir = os.path.dirname(os.path.realpath(__file__))

supported_types = ['PLAY', 'TRAIN', 'TEST', 'EVALUATE']

def correct_Punctuation(supported_Types):
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

def main(typeGame):
   '''Prompts user for what game then runs it'''

   try:
      typeGame = sys.argv[1]
   except:
      typeGame = None
   
   try:
      fps = int(sys.argv[2])
   except:
      fps = 30
   
   try:
      epoch = int(sys.argv[3])
   except:
      if typeGame != None and typeGame.upper() == 'EVALUATE':
         epoch = None
      else:
         epoch = 2500


   if typeGame == None:
      typeGame = input('Type of Game? ')
      while True:  
         if typeGame.upper() not in supported_types:
            print('Only types of games supported are: ' + correct_Punctuation(supported_types))
            typeGame = input('Type of Game? ')
         else:
            break
   

   if typeGame.upper() == 'TRAIN':
      gm = fp.gameManager(TrainGame(file_dir = file_dir))
      agent = fm.agent(gm)
      fm.train(agent, epochs = epoch, plotting_scores=True)

   elif typeGame.upper() == 'PLAY':
      gm = fp.gameManager(PlayGame(file_dir = file_dir))
      gm.play()

   elif typeGame.upper() == 'EVALUATE':
      gm = fp.gameManager(EvaluateGame(file_dir = file_dir, fps = fps))
      agent = fm.agent(gm)
      agent.model.load_state_dict(torch.load(file_dir + '/test_weights.pt'))
      fm.evaluate(agent)

   elif typeGame.upper() == 'TEST':

      gm = fp.gameManager(TrainGame(file_dir = file_dir))
      agent = fm.agent(gm)
      fm.train(agent,epochs=epoch) #epochs here are the number of games run in order to trains
      gm = fp.gameManager(EvaluateGame(file_dir = file_dir))
      agent.new_gm(gm)
      print('starting to evaluate')
      for i in range(3): #number of times it is going to be evaluated
         fm.evaluate(agent, run_num = i, epochs = 250) #epochs here are the number of games used when evaluating 
         


if __name__ == "__main__":
   main(None)

