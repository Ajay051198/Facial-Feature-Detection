import tensorflow as tf
import load_data
import build_model
import matplotlib.pyplot as plt

trainX, trainy = load_data.prepare_data('train')
model = build_model.CNN()



#Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint('CNN_facefeatures.h5', monitor='mae', verbose=0, save_best_only=True,
                                                save_weights_only=False, mode='auto', save_freq='epoch')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mae', min_delta=0, patience=20, verbose=1, mode='auto')

callback_list = [early_stopping]

EPOCHS = 500

history = model.fit(trainX,
          trainy,
          epochs=EPOCHS,
          callbacks=callback_list)

model.save('CNN_facefeatures.h5')
model.save_weights('CNN_facefeatures_weights.h5')

plt.plot(history.history['mae'])
plt.title('Model performance')
plt.ylabel('Mean absolute error')
plt.xlabel('Epoch')
plt.show()
plt.savefig('model_performance.png')
