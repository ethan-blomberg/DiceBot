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

for j in range(20):
    path = directory + '/' + str(j+1) 
    os.chdir(path)
    files = glob.glob(path + '/*.png')   

    for k in range(len(files)):
        img = cv.imread(files[k])

        for l in range(3):
            #augment
            (rows, cols) = img.shape[:2]
            Mat = cv.getRotationMatrix2D((cols / 2, rows / 2), (90*(l+1)), 1)
            res = cv.warpAffine(img, Mat, (cols, rows))

            ## Store Image in Labeled Folder
            try: 
                file_num = get_next_num(j+1)
            except:
                file_num = 0

            filename = str(file_num) + '.png'
            cv.imwrite(filename,res)
    print('Completed ' + str(j+1))

print('\nDone!\n')