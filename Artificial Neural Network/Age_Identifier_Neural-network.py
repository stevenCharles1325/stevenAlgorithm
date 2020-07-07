## This might be the Silliest program of all Lol
## Obviously, This can be done without using artifical Neural Network
## But just for the sake of my little practice please try
## Thank you Very much : )

import numpy as np
import numpy.random as rn
#from sklearn.metrics import accuracy_score

def f(x):
  return 1 /(1 + np.exp(-x))
  
def f_deriv(x):
  return f(x) * (1-f(x))

def nnStructure(inpL, hidL, outL):
  # initialize weights and bias in Input and hidden layer  
  weights_IH = rn.random_sample((hidL, inpL))  
  bias_IH =  rn.random_sample((hidL, 1))
  
  # initializing weights and bias in Input and hidden layer
  weights_HO =  rn.random_sample((outL, hidL))
  bias_HO =  rn.random_sample((outL, 1))
  
  return weights_IH, bias_IH, weights_HO, bias_HO
  
wIH, bIH, wHO, bHO = nnStructure(4,8,4)

Lrate = 0.01
def trainTime(input_x, targetData):
  global wHO, wIH, bHO, bIH

  # feed forward process
  inputs_x = input_x[:,np.newaxis]
  target = targetData[:,np.newaxis]
  hidden = np.dot(wIH, inputs_x)
  hidden = np.add(hidden, bIH)
  hidden = f(hidden)

  yOutput = np.dot(wHO, hidden)
  yOutput = np.add(yOutput, bHO)
  yOutput = f(yOutput)
 
  ## back progpagation and gradient descent
  # calculate output's error
  Oerror = np.subtract(target,yOutput)

  #gradient for the output
  outputGradient = f_deriv(yOutput)
  outputGradient = np.multiply(outputGradient, Oerror)
  outputGradient = np.multiply(outputGradient, Lrate)
  
  #delta weight in hidden and output
  deltaWHO = np.dot(outputGradient, np.transpose(hidden))
   
  wHO = np.add(wHO, deltaWHO)
 
  #for bias in hidden and output layer
  bHO = np.add(bHO, outputGradient)
  
  #hidden layer Error
  Herror = np.dot(np.transpose(wHO), Oerror)
  
  # gradient of hidden and input
  hiddenGradient = f_deriv(hidden)
  hiddenGradient = np.multiply(hiddenGradient, Herror)
  hiddenGradient = np.multiply(hiddenGradient, Lrate)
  

  #delta weights in input and hidden layer
  deltaWIH = np.dot(hiddenGradient, np.transpose(inputs_x))
  
  wIH = np.add(wIH, deltaWIH)
 
  #for bias in input and hidden layer
  bIH = np.add(bIH, hiddenGradient)  
  


def feed_forward(inputs_x):
  
  input = inputs_x[:,np.newaxis]
  hidden = np.dot(wIH, input)
  hidden = np.add(hidden, bIH)
  hidden = f(hidden)
  #print(hidden)
  yOutput = np.dot(wHO, hidden)
  yOutput = np.add(yOutput, bHO)
  yOutput = f(yOutput)

  return yOutput.flatten()

# Converting the target value to vector
def targetClassif(input):
  if input in range(0,13):
    return np.array([1,0,0,0])
    
  elif input in range(13,19):
    return np.array([0,1,0,0])
    
  elif input in range(19,65):
    return np.array([0,0,1,0])
    
  elif input >= 65:
    return np.array([0,0,0,1])

print('[Please wait while doing some training]')
print('[Please dont do anything]')
for iteration in range(500):
  for i in range (1,101):
    #training set
    trainTime(targetClassif(i),
    targetClassif(i))
 
  
for i in range(5):
  a = int(input('Enter age: '))
  outArr =feed_forward(targetClassif(a))
  outArr = np.argmax(outArr)
  if outArr == 0:
    print('>>> Child')
  elif outArr == 1:
    print('>>> Teenager')
  elif outArr == 2:
    print('>>> Adult')
  elif outArr == 3:
    print('>>> Senior')
  print('')

  
  
