import dataset
import colorizer_cnn
import colorizer_gan
import tensorflow as tf
import wandb
tf.config.run_functions_eagerly(True)

'''
model = colorizer_cnn.colorizer_cnn()
model.load()
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.00001))
#print('processing data...')
#dataset.process_data()
#print('loading data...')
#print('training model...')
model.train(train_ds,valid_ds,output_file='u-net-colorizer-v-1.h5',epochs_=5)
model.test(test_ds=test_ds)
model.colorize(['Data/processed/unlabeled_image_png_17544.png'],'results')
'''

(train_ds,valid_ds,test_ds) = dataset.load_ds()
gan = colorizer_gan.colorizer_gan()
gan.create()
gan.train(train_ds=train_ds,valid_ds=valid_ds,epochs=5)
gan.evaluate(test_ds,is_validate=False)

