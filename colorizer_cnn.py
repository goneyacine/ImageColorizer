from numpy import iterable
import tensorflow as tf
import os
import cv2

def euclidean_distance_loss(y_true,y_pred):
        return (1/(32**2)) * tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))
    
def PSNR(y_true, y_pred):
     max_pixel = 1.0
     return (10.0 * tf.keras.backend.log((max_pixel ** 2) / (tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1)))) / 2.303
    
class colorizer_cnn:
    
    def __init__(self) -> None:
       pass
    
    
    def load(self,model_path='u-net-colorizer.h5'):
        self.model = tf.keras.models.load_model(model_path
                     ,custom_objects={'euclidean_distance_loss':euclidean_distance_loss,'PSNR':PSNR})
        
    def save(self,output_file='u-net-colorizer.h5'):
        if self.model == None:
            print('Error : model in not initialized')
            return
        print('saving model...')
        self.model.save(output_file)
        print('model saved')
    def create(self):
        #encoder
        input  = tf.keras.Input(shape=(32,32,1))
        conv1 = tf.keras.layers.Conv2D(64,(4,4),strides=(2,2),padding='same')(input)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.LeakyReLU(0.2)(conv1)
        conv2 = tf.keras.layers.Conv2D(128,(4,4),strides=(2,2),padding='same')(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.LeakyReLU(0.2)(conv2)
        conv3 = tf.keras.layers.Conv2D(256,(4,4),strides=(2,2),padding='same')(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.LeakyReLU(0.2)(conv3)
        conv4 = tf.keras.layers.Conv2D(512,(4,4),strides=(2,2),padding='same')(conv3)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.LeakyReLU(0.2)(conv4)
    
        
        #decoder
        deconv1 = tf.keras.layers.Conv2DTranspose(256,(4,4),strides=(2,2),activation='relu',padding='same')(conv4)
        deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
        deconv1 = tf.keras.layers.ReLU()(deconv1)
        deconv1 = tf.concat([conv3,deconv1],axis=3)
        deconv2 = tf.keras.layers.Conv2DTranspose(128,(4,4),strides=(2,2),activation='relu',padding='same')(deconv1)
        deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
        deconv2 = tf.keras.layers.ReLU()(deconv2)
        deconv2 = tf.concat([conv2,deconv2],axis=3)
        deconv3 = tf.keras.layers.Conv2DTranspose(64,(4,4),strides=(2,2),activation='relu',padding='same')(deconv2)
        deconv3 = tf.keras.layers.BatchNormalization()(deconv3)
        deconv3 = tf.keras.layers.ReLU()(deconv3)
        deconv3 = tf.concat([conv1,deconv3],axis=3)
        
        
        output = tf.keras.layers.Conv2DTranspose(3,(4,4),strides=(2,2),activation=tf.keras.activations.tanh,padding='same')(deconv3)
        self.model = tf.keras.models.Model(inputs=input,outputs=output)
        print(self.model.summary())
        
    def compile(self,optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001),loss=euclidean_distance_loss,metrics=['accuracy',PSNR]):
        self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        
    def train(self,train_ds,valid_ds,epochs_=10,batch_size_=4000,save_model=True,output_file='u-net-colorizer.h5'):
        if self.model == None:
            print('Error : model is not initialzed.')
            return
        for i in range(epochs_):
         print('epoch ' +str(i+1))
         for x,y in train_ds.batch(batch_size_):
            self.model.fit(x,y,validation_data=valid_ds,batch_size=64,epochs=1)
        if save_model:
           print('saving model...')
           self.model.save(output_file)
           print('model saved.')
    def test(self,test_ds):
        return self.model.evaluate(test_ds)
    
    def colorize(self,files,output_directory):
        if self.model == None:
            print('Error : model is not initialzed.')
            return
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        
        ds = []
        for file in files:
         img = cv2.imread(file)
         img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype('float32') / 255
         ds.append(img)
        
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.batch(1)
        results = self.model.predict(ds)
        print(results[0][:,:,0])
        for i in range(len(results)):
            cv2.imwrite(os.path.join(output_directory,str(i) +'.jpeg' ),results[i] * 255)
        