from flappy_bird import FlappyPlayGame, FlappyEvaluateGame, FlappyTrainGame, FlappyGameManager
import flappy_model as fm
import torch
import os 
import sys

SUPPORTED_TYPES = ['PLAY', 'TRAIN', 'TEST', 'EVALUATE']

def main() -> None:
   '''Runs the main program that will either take in command line arguements or will prompt the user'''
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
         if type_game.upper() not in SUPPORTED_TYPES:
            print('Only types of games supported are: ' + _correct_punctuation(SUPPORTED_TYPES))
            type_game = input('Type of Game? ')
         else:
            break
   

   if type_game.upper() == 'TRAIN':
      gm = FlappyGameManager(FlappyTrainGame(), game_type='train')
      agent = fm.Agent(gm)
      fm.train(agent, epochs = epoch, plotting_scores=True)

   elif type_game.upper() == 'PLAY':
      gm = FlappyGameManager(FlappyPlayGame())
      gm.play()

   elif type_game.upper() == 'EVALUATE':
      gm = FlappyGameManager(FlappyEvaluateGame(), game_type='evaluate')
      agent = fm.Agent(gm)
      agent.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'test_weights.pt')))
      fm.evaluate(agent, epochs = epoch)

   elif type_game.upper() == 'TEST':

      gm = FlappyGameManager(FlappyTrainGame())
      agent = fm.Agent(gm)
      fm.train(agent,epochs=epoch) #epochs here are the number of games run in order to trains
      gm = FlappyGameManager(FlappyEvaluateGame())
      agent.gm = gm
      print('starting to evaluate')
      for i in range(3): #number of times it is going to be evaluated
         fm.evaluate(agent, run_num = i, epochs = 250) #epochs here are the number of games used when evaluating 
         

def _correct_punctuation(supported_types : list[str]) -> str:
   '''Returns a string with the support types in the correct punctuation'''
   if len(supported_types)>2:
      string = ", ".join(supported_types)
   else:
      string = " ".join(supported_types)

   string = string.lower()
   string = string.title()
   index_of_last_word = len(string) - len(supported_types[-1])
   string = string[:index_of_last_word] + 'and ' + string[index_of_last_word:]

   return string

if __name__ == "__main__":
   main()