
from FlappyBird import flappy as fp
import flappyModel as fm

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
   gm = fp.gameManager(typeGame)
   if typeGame.upper() == 'TRAIN':
      agent = fm.agent(typeGame)
      fm.train(agent)
   elif typeGame.upper() == 'PLAY':
      gm.play()
   elif typeGame.upper() == 'TEST':
      agent = fm.agent(typeGame)
      fm.train(agent,epochs=10000) #epochs here are the number of games run in order to train
      print('starting to evaluate')
      for i in range(3): #number of times it is going to be evaluated
         fm.evaluate(agent, run_num = i, epochs = 1000) #epochs here are the number of games used when evaluating 
         






main(None)
   





