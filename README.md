# EEG_decoding

To use in Google Colab : 

    !pip install mne==1.3.1
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    from google.colab import files
    uploaded = files.upload() # run and select Data_processing.py 

    from Data_processing import *

    file_path = '/content/drive/MyDrive/Colab_Notebooks/data/resampled_epochs_subj_0.pkl' #Select the folder where data is

    epochs, labels = get_data(file_path)