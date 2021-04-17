'''
    dataset_generator is used to prepare the images for the keras classifier
    to read the images in the correct form.
    
    No preprocessing is done here, just preparing the directories.
    
    Keep the dataset GTSRB_Classification.zip inside the main's dir.
'''


class createDataBase():
    def dataPrepare():
        from os import walk
        import csv
        import os.path
        from shutil import copyfile
        import zipfile
        
        # Unzipping the file
        if os.path.isdir('GTSRB') == True:
            try:
                with zipfile.ZipFile("GTSRB_Classification.zip","r") as zip_ref:
                    zip_ref.extractall()
            except FileNotFoundError:
                print('File GTSRB_Classification not found!')
            

        path = 'GTSRB/'
        
        _, _, filenames = next(walk(path))
        filenames = sorted(filenames)
        numclass = 42
        
        if (os.path.isdir('dataset') == False):
            os.mkdir('dataset')
        for i in range(numclass):
            if (os.path.isdir('dataset/' + str(i)) == False):
                os.mkdir('dataset/' + str(i))
        
        
        files = []
        for i in range(numclass):
            for j in range (len(filenames)):
                image = str(filenames[j])
                image_class = image.split('_')[0]
                if (image_class == str(i)):
                    files.append([image_class, image])
                    copyfile(path + image, 'dataset/' + image_class + '/' + image)
                    
        with open('dataset.csv', mode='w') as file:
            file_writer = csv.writer(file, delimiter=',', quotechar='"', 
                                     quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['Class', 'Instance'])
            for i in range (len(files)):
                file_writer.writerow([files[i][0], files[i][1]])
        file.close()
        


