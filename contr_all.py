import os
import cv2 as cv
import glob

################################
directory = 'C:/DiceStuff/DataSortedTF'
################################

def get_next_num(Directory):
    filelist = [filename for filename in os.listdir(path) \
        if filename]
    for i in range(len(filelist)):
        filelist[i] = int(''.join(filelist[i].split())[:-4])
        
    return max(filelist) + 1

# change to 20
for j in range(20):
    path = directory + '/' + str(j+1) 
    os.chdir(path)
    files = glob.glob(path + '/*.png')   

    for k in range(len(files)):
        img = cv.imread(files[k])
        
        #augment
        res = cv.convertScaleAbs(img,alpha=0.5,beta=0)

        ## Store Image in Labeled Folder
        try: 
            file_num = get_next_num(j+1)
        except:
            file_num = 0

        filename = str(file_num) + '.png'
        cv.imwrite(filename,res)
    print('Completed ' + str(j+1))
print('\nDone!\n')