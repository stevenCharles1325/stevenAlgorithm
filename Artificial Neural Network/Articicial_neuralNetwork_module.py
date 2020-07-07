import numpy as np
import numpy.random as rn
from sklearn.metrics import accuracy_score

class neuralNetwork:
  
  learningRate = 0.1
  wantPredict = True 
  def __init__(self, iLayer, hLayer, oLayer):
    self.iLayer = iLayer
    self.hLayer = hLayer
    self.oLayer = oLayer
    
    # weights and bias for input and hidden layer
    self.weightsIH = rn.random_sample((self.hLayer, self.iLayer))
    self.weightsHO = rn.random_sample((self.oLayer, self.hLayer))
    
    self.biasIH =  rn.random_sample((self.hLayer, 1))
    self.biasHO =  rn.random_sample((self.oLayer, 1))
  
  def oneHot(self, target):
    trgt = np.zeros((10,1))
    trgt[target] = 1
    return trgt

  # sigmoid function --> activation function  
  def f(self,x):
    return 1 / (1+np.exp(-x))

  # derivative of sigmoid function  
  def fDeriv(self, x):
    return self.f(x) * (1-self.f(x))
  
  @classmethod
  def setlearningRate(cls, value):
    cls.learningRate = value

  def setWantPredict(self, bool):
    self.wantPredict = bool

  # feed forward algorithm
  def feedForward(self, input):
    
    input = input[:,np.newaxis]
    
    hidden = np.dot(self.weightsIH,input)
    hidden = np.add(hidden, self.biasIH)
    hidden = self.f(hidden)
    
    output = np.dot(self.weightsHO, hidden)
    output = np.add(output, self.biasHO)
    output = self.f(output)
    if self.wantPredict:
      return np.argmax(output.flatten())
    else:
      return output, hidden, input
      
  # algo for training and gradient descent  & backpropagation  
  def trainAlgo(self, inputData, targetData):
    global weightsIH,biasIH,weightsHO,biasHO,wantPredict
    self.wantPredict = False
    
    output, hidden, input = self.feedForward(inputData)
    
    target = self.oneHot(targetData)
    
    # calculate the output error
    Oerror = np.subtract(target, output)
    
    gradientHO = self.fDeriv(output)
    gradientHO = np.multiply(gradientHO,Oerror)
    gradientHO = np.multiply(gradientHO,self.learningRate)
    
    # calculate delta weights in hidden & output
    deltaWHO = np.dot(gradientHO,np.transpose(hidden))
    
    # update weights and bias
    self.weightsHO = np.add(self.weightsHO,deltaWHO)

    self.biasHO = np.add(self.biasHO, gradientHO)
    
    Herror = np.dot(np.transpose(self.weightsHO), Oerror)

    # calculate gradient in input & hidden layer
    gradientIH = self.fDeriv(hidden)
    gradientIH = np.multiply(gradientIH,Herror)
    gradientIH = np.multiply(gradientIH,self.learningRate)
    
    # calculate delta weights in input & output
    deltaWIH = np.dot(gradientIH,np.transpose(input))
    
    # update weights and bias
    self.weightsIH = np.add(self.weightsIH, deltaWIH)
    self.biasIH = np.add(self.biasIH, gradientIH)
    
  def predictIt(self, xtest):
    #print(xtest)
    m = xtest.shape[0]
    y = np.zeros((m,))
    for i in range(m):
      y[i] = self.feedForward(xtest[i])
      
    return y
    
  @staticmethod
  def accuracyIs(yTest, targetPred):
    return accuracy_score(yTest, targetPred) * 100
  
  @classmethod
  def saveTrain(cls,bool=True):
    if bool:
      np.save('nnWIH.npy',weightsIH)
      np.save('nnWHO.npy',weightsHO)
      np.save('nnBIH.npy',biasIH)
      np.save('nnBHO.npy',biasHO)
      
    # load neural network prev training
    cls.wieghtsIH = np.load('nnWIH.npy')
    cls.wieghtsHO = np.load('nnWHO.npy')
    cls.biasIH = np.load('nnBIH.npy')
    cls.biasHO = np.load('nnBHO.npy')
    
    
    
    
    
    
    
    
    
