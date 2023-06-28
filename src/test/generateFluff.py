import random
import re



def main():
    with open('fluff.txt','r') as rw:
        fluff = rw.readline()
        fluffTexts = re.split('!_!',fluff)    
        for x in range(len(fluffTexts)):
             with open(f'{x}.txt','w') as w:
                w.write(fluffTexts[x])
    


if __name__=='__main__':
    main()