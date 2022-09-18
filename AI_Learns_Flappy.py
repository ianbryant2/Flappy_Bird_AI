
from FlappyBird import flappy as fp
from FlappyBird.flappy import PlayGame, TrainGame, EvaluateGame
import flappyModel as fm
import torch

supported_types = ['PLAY', 'TRAIN', 'TEST']

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
   if typeGame == None:
      typeGame = input('Type of Game? ')
      while True:  
         if typeGame.upper() not in supported_types:
            print('Only types of games supported are: ' + correct_Punctuation(supported_types))
            typeGame = input('Type of Game? ')
         else:
            break
   

   if typeGame.upper() == 'TRAIN':
      gm = fp.gameManager(TrainGame())
      agent = fm.agent(gm)
      fm.train(agent)

   elif typeGame.upper() == 'PLAY':
      gm = fp.gameManager(PlayGame())
      gm.play()

   elif typeGame.upper() == 'EVALUATE':
      gm = fp.gameManager(EvaluateGame(fps=30))
      agent = fm.agent(gm)
      agent.model.load_state_dict(torch.load("test_weights.pt"))
      fm.evaluate(agent)

   elif typeGame.upper() == 'TEST':

      gm = fp.gameManager(TrainGame())
      agent = fm.agent(gm)
      fm.train(agent,epochs=2500) #epochs here are the number of games run in order to trains
      gm = fp.gameManager(EvaluateGame())
      agent.new_gm(gm)
      print('starting to evaluate')
      for i in range(3): #number of times it is going to be evaluated
         fm.evaluate(agent, run_num = i, epochs = 250) #epochs here are the number of games used when evaluating 
         


if __name__ == "__main__":
   main(None)

