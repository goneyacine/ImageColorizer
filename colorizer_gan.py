import numpy as np
import tensorflow as tf
import os
import cv2
import wandb
import datetime

class colorizer_gan:
    def __init__(self,d_lr=1e-4,g_lr=2e-4,batch_size=128):
         self.d_lr = d_lr
         self.g_lr = g_lr
         self.batch_size = batch_size
         self.init_losses()
         self.create_optimizers()
         
         
        
    def init_generator(self):
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
        return tf.keras.models.Model(inputs=input,outputs=output)
    
    def init_discriminator(self):
        input  = tf.keras.Input(shape=(32,32,4))
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
        output = tf.keras.layers.Conv2D(1,(4,4),strides=(2,2),padding='same',activation=None)(conv4)
        return tf.keras.models.Model(inputs=input,outputs=output)
    
    def init_model(self):
        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
    
        
    def train(self,train_ds,valid_ds,epochs=1,auto_save=True,output_path='gan colorizer',colorizing_samples=None,colorize_samples=True):
        wandb.init(
            # set the wandb project where this run will be logged
            project="ImageColorizer",
            
            # track hyperparameters and run metadata
            config={
            "g_learning_rate": self.g_lr,
            "d_learning_rate": self.d_lr,
            "batch_size":self.batch_size,
            "architecture": "gan",
            "epochs": epochs,
            "train_data_size": len(train_ds),
            "valid_data_size": len(valid_ds)
            }
        )
        @tf.function
        def step(x,y):
            real_input = tf.concat([x[...,None], y], axis=-1)
            with tf.GradientTape() as r_tape:
             d_real_loss = self.loss_d_real(self.discriminator(real_input,training=True))
            d_r_gradients = r_tape.gradient(d_real_loss,
                                            self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_r_gradients,
                                                 self.discriminator.trainable_variables))
            
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                 fake_image = self.generator(x, training=True)
                 fake_input = tf.concat([x[...,None], fake_image], axis=-1)
                 fake_logits = self.discriminator(fake_input, training=True)
                 d_fake_loss = self.loss_d_fake(fake_logits)
                 g_loss = self.loss_g(y, fake_image,fake_logits)
                 psnr = tf.image.psnr(fake_image,y,max_val= 1.0) 
            d_f_gradients = d_tape.gradient(d_fake_loss,
                                            self.discriminator.trainable_variables)
            g_gradients = g_tape.gradient(g_loss,
                                          self.generator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_f_gradients,
                                                 self.discriminator.trainable_variables))
            self.g_optimizer.apply_gradients(zip(g_gradients,
                                                 self.generator.trainable_variables))
            print(f'psnr {tf.reduce_mean(psnr).numpy()} g_loss {g_loss} d_fake_loss {d_fake_loss} d_real_loss {d_real_loss}')
            wandb.log({'psnr ':tf.reduce_mean(psnr).numpy(),"g_loss" :g_loss ,'d_fake_loss':d_fake_loss,'d_real_loss':d_real_loss})
                
            return (psnr,g_loss,d_fake_loss,d_real_loss)
        print('training...')
        for i in range(epochs):
            print(f'...epoch {i+1}...')            
            for x,y in train_ds.batch(self.batch_size):
                step(x,y)
            self.evaluate(valid_ds)   
            if auto_save:
             print('auto saving model...')
             self.save(output_path=output_path) 
             print('auto saving done...') 
             if colorize_samples:
                 self.colorize(colorizing_samples[0],colorizing_samples[1],output_folder='gan_results/'.join(str(i)))
        print('training finished...')    
             
    
    def loss_g(self,real_img,fake_img,fake_logits,gamma=100):
        bce_loss = tf.reduce_mean(self.bce(tf.ones_like(fake_logits),
                                           fake_logits),axis=[1,2])
        l1_loss = tf.reduce_mean(tf.abs(real_img - fake_img), axis=[1, 2, 3])
        return tf.nn.compute_average_loss(bce_loss + gamma * l1_loss,
                                          global_batch_size=self.batch_size)
    
    def loss_d_real(self,real_logits):
        real_loss = tf.reduce_mean(self.bce_smooth(tf.ones_like(real_logits),
                                                   real_logits), axis=[1, 2])
        return tf.nn.compute_average_loss(real_loss,
                                          global_batch_size=self.batch_size)
    def loss_d_fake(self,fake_logits):
        fake_loss = tf.reduce_mean(self.bce(tf.zeros_like(fake_logits),
                                            fake_logits), axis=[1, 2])
        return tf.nn.compute_average_loss(fake_loss,
                                          global_batch_size=self.batch_size)
    
    def create_optimizers(self):
            self.d_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.d_lr, beta_1=0.5)
            self.g_optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.g_lr, beta_1=0.5)
    def init_losses(self):
        self.bce_smooth = tf.losses.BinaryCrossentropy(from_logits=True,
                                                                 label_smoothing=0.1,
                                                                 reduction=tf.keras.losses.Reduction.NONE)
        self.bce =  tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.NONE)
    
    def save(self,output_path='gan colorizer'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.generator.save(os.path.join(output_path,'generator.keras'))
        self.discriminator.save(os.path.join(output_path,'discriminator.keras'))
    
    def evaluate(self,ds,is_validate=True):
        d_real_loss = np.empty(0)   
        d_fake_loss = np.empty(0)
        g_loss = np.empty(0)
        psnr = np.empty(0)

        for x,y in ds.batch(1):
            real_input = tf.concat([x[...,None], y], axis=-1)
            d_real_loss = np.append(d_real_loss,self.loss_d_real(self.discriminator(real_input,training=False)))
            fake_image = self.generator(x, training=False)
            fake_input = tf.concat([x[...,None], fake_image], axis=-1)
            fake_logits = self.discriminator(fake_input, training=False)
            d_fake_loss = np.append(d_fake_loss,self.loss_d_fake(fake_logits))
            g_loss = np.append(g_loss,self.loss_g(y, fake_image,fake_logits))
            psnr = np.append(psnr,tf.image.psnr(fake_image,y,max_val= 1.0))
        if is_validate:
         print(f'val_psnr {np.mean(psnr)} val_g_loss {np.mean(g_loss)} val_d_fake_loss {np.mean(d_fake_loss)} val_d_real_loss {np.mean(d_real_loss)}')     
         wandb.log({'val_psnr':np.mean(psnr),"val_g_loss" :np.mean(g_loss) ,'val_d_fake_loss':np.mean(d_fake_loss),'val_d_real_loss':np.mean(d_real_loss)})   
        else:
         print(f'psnr {np.mean(psnr)} g_loss {np.mean(g_loss)} d_fake_loss {np.mean(d_fake_loss)} d_real_loss {np.mean(d_real_loss)}')
    
    def load_model(self,folder_path='gan colorizer'):
        self.generator = tf.keras.models.load_model(os.path.join(folder_path,'generator.keras'))
        self.discriminator = tf.keras.models.load_model(os.path.join(folder_path,'discriminator.keras'))
    
    def colorize(self,gray_img,true_img,output_folder='gan_results',compare_to_original=True):
        if self.generator == None or self.discriminator == None:
            print('Error : model is not initialzed.')
            return
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        ds = tf.data.Dataset.from_tensor_slices(gray_img)
        ds = ds.batch(1)
        results = self.generator.predict(ds)
        for i in range(len(results)):
           if compare_to_original:
            cv2.imwrite(os.path.join(output_folder,str(i) +'.jpeg' ),cv2.hconcat([results[i] * 255,true_img[i]]))
           else :
            cv2.imwrite(os.path.join(output_folder,str(i) +'.jpeg' ),results[i] * 255)