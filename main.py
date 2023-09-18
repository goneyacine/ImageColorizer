import dataset
import colorizer_cnn
import tensorflow as tf

model = colorizer_cnn.colorizer_cnn()
model.load()
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.00001))
#print('processing data...')
#dataset.process_data()
#print('loading data...')
(train_ds,valid_ds,test_ds) = dataset.load_ds()
#print('training model...')
model.train(train_ds,valid_ds,output_file='u-net-colorizer-v-1.h5',epochs_=5)
model.test(test_ds=test_ds)
model.colorize(['Data/processed/unlabeled_image_png_17544.png'],'results')


