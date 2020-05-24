import matplotlib.pyplot as plt
import load_data
import tensorflow as tf
from random import randint

X, y = load_data.prepare_data('validation')
model = tf.keras.models.load_model('assets/CNN_facefeatures.h5')
model.load_weights('assets/CNN_facefeatures_weights.h5')
print(model.summary())
print(X.shape)
print(y.shape)
predictions = model.predict(X, verbose=1)
print('x')
print(predictions.shape)

#idx = 1030

'''
#training set
f,ax = plt.subplots(8,8,figsize=(20,20))
f.subplots_adjust(0,0,2,2)
for i in range(0,8,1):
    for j in range(0,8,1):
        idx = randint(0,len(X))
        X_temp = X[idx]
        y_temp = y[idx]
        predictions_temp = predictions[idx]
        ax[i,j].imshow(X_temp.reshape(96,96), cmap='gray')
        for k in range(1,31,2):
            ax[i,j].plot(y_temp[k-1], y_temp[k], 'ro')
            ax[i,j].plot(predictions_temp[k-1], predictions_temp[k], 'bx')
        ax[i,j].axis('off')
plt.show()
'''

#testset
f,ax = plt.subplots(8,8,figsize=(20,20))
f.subplots_adjust(0,0,2,2)
for i in range(0,8,1):
    for j in range(0,8,1):
        idx = randint(0,len(X))
        X_temp = X[idx]
        predictions_temp = predictions[idx]
        ax[i,j].imshow(X_temp.reshape(96,96), cmap='gray')
        for k in range(1,31,2):
            ax[i,j].plot(predictions_temp[k-1], predictions_temp[k], 'bx')
        ax[i,j].axis('off')
plt.show()
