
from FlapPyBird import flappy as fp
import flappyModel as fm

supported_types = ['PLAY', 'TRAIN']

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
      agent.train()
   elif typeGame.upper() == 'PLAY':
      gm.play()  




main('train')
   





