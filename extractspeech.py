

# Import Module
import os
import re

  
# Folder Path
#file_path = "C:/VKB/Chindu_Project/DataSet/Training/NCS"

file_path = "C:/VKB/Chindu_Project/DataSet/Training/ACS"


# Read list of files in the directory

os.chdir(file_path)
#outFile = open('C:/VKB/Chindu_Project/DataSet/ncs.txt', 'a')
outFile = open('C:/VKB/Chindu_Project/DataSet/newacs.txt', 'a')
for path in os.scandir(file_path):
    
   
    if path.is_file():
                 
        shakes = open(path.name)
        print(path.name)
        for line in shakes:   
           
            if re.match("(.*)CHI:(.*)", line):
                
                outFile.write(line)
                #print(line)
      
                

    
outFile.close()