import numpy as np
from PIL import Image
import tensorflow as tf


# Custom data generator for model training

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(256, 256, 1),
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle 
        self.n = len(self.df)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, image_id, target_size):
        image = Image.open(image_id).convert('RGB')
        image = image.resize((target_size[0], target_size[1]), Image.ANTIALIAS)
        image = np.array(image)
        return image/255
    
    def __get_output(self, image_id, target_size):
        image = Image.open(image_id).convert('RGB')
        image = image.resize((target_size[0], target_size[1]), Image.ANTIALIAS)
        image = np.array(image)
        return image/255
    
    def __get_data(self, batches):
        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in batches[self.X_col]])
        y_batch = np.asarray([self.__get_output(y, self.input_size) for y in batches[self.y_col]])
        return X_batch, y_batch

    def __get_one_data(self, row):
        X = np.asarray(self.__get_input(row[self.X_col], self.input_size))
        y = np.asarray(self.__get_output(row[self.y_col], self.input_size))
        return X, y

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y

    def get_all_samples(self):
        X, y = self.__get_data(self.df)        
        return X, y

    def get_sample(self, index):
        X, y = self.__get_one_data(self.df.loc[index])        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size