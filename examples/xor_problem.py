from library.neuralNetwork import *

nn = NeuralNetwork(2,3,1)
training_epoch = 1000
img_size = 100

for i in range(training_epoch):
   if(i%4 == 0):
      nn.train(np.array([[1],[0]]),np.array([[1]]))
   elif(i%4 == 1):
      nn.train(np.array([[0],[1]]),np.array([[1]]))
   elif(i%4 == 2):
      nn.train(np.array([[0],[0]]),np.array([[0]]))
   else:
      nn.train(np.array([[1],[1]]),np.array([[0]]))

im = np.zeros((img_size,img_size))
for x in range(img_size):
   for y in range(img_size):
      im[x][y] = nn.predict(np.array([[x/img_size], [y/img_size]]))[0][0]
plt.imshow(im)
plt.show()