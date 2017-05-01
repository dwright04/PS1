""" Sparse Filtering """

import sys
import numpy as np
import scipy.io as sio

class sparse_filtering(object):
  """Sparse filtering algorithm due to Ngiam et al. 2011.
    
     https://papers.nips.cc/paper/4334-sparse-filtering
     
     Parameters
     ----------
     k : int, optional, default 256
         The number of data representations to learn.
        
     max_iter : int, optional, default 200
         The maximum number of iterations of the sparse filtering algorithm.

  """
  def __init__(self, k=256, max_iter=200):
    self.k = int(k)
    self.max_iter = int(max_iter)

  def fit(self, X, channels=1):
    """
       Parameters
       ----------
       X : array, shape = [n_features, n_samples]
           Array representing the data.
       
       channels : int, optional, default 1
           The number of image channels. RGB = 3, grayscale = 1.
    """
    from scipy import optimize
    
    def objective(W, *args):
      X, pad = args
      Obj, DeltaW = self.objective(X, W)
      return Obj
    
    def objectiveG(W, *args):
      X, pad = args
      Obj, DeltaW = self.objective(X, W)
      return DeltaW
    
    self.iteration = 0
    self.trained_W = None
    self.channels = int(channels)
    self.n_features, m = np.shape(X)
    args = (X, 1)
    initW = np.ravel(np.random.rand(self.k, self.n_features), order="F")
    optW = optimize.fmin_cg(objective, \
                            x0 = initW, \
                            fprime = objectiveG, \
                            args = args, \
                            max_iter = self.max_iter, \
                            callback = self.callback)
    self.trained_W = optW

  def objective(self, X, W, channels):
    """
    """
    n, m = np.shape(X)
    W = np.reshape(W, (self.k, n), order="F")
    # Feed forward
    F = np.dot(W, X)
    Fs = np.sqrt(np.multiply(F, F) + 1e-8)
    NFs, L2Fs = self.l2row(Fs)
    Fhat, L2Fn = self.l2row(NFs.T)
    # Compute objective function
    Obj = self.objectiveFunc(Fhat)
    # Backpropagate through each feed forward step
    DeltaW = self.l2rowg(NFs.T, Fhat, L2Fn, np.ones(np.shape(Fhat)))
    DeltaW = self.l2rowg(Fs, NFs, L2Fs, DeltaW.T)
    DeltaW = np.dot(np.multiply(DeltaW, (F/Fs)), X.T)
    DeltaW = np.ravel(DeltaW, order="F")
    return Obj, DeltaW

  def l2row(self, X):
    """
    """
    n, m = np.shape(X)
    N = np.sqrt(np.sum(np.multiply(X, X), axis=1) + 1e-8)
    N_stack = np.tile(N, (m, 1)).T
    Y = np.divide(X, N_stack)
    return Y, N

  def objectiveFunc(self, Fhat):
    """
    """
    return np.sum(Fhat)

  def l2rowg(self, X, Y, N, D):
    """
    """
    n, m = np.shape(X)
    N_stack = np.tile(N, (m, 1)).T
    firstTerm = np.divide(D, N_stack)
    sum = np.sum(np.multiply(D, X), 1)
    sum = sum / (np.multiply(N,N))
    sum_stack = np.tile(sum[np.newaxis], (np.shape(Y)[1],1)).T
    secondTerm = np.multiply(Y, sum_stack)
    return firstTerm - secondTerm

  def feedForward(self, W, X):
    """
    """
    # Feed Forward
    n, m = np.shape(X)
    W = np.reshape(W, (self.k, self.n_features), order="F")
    F = np.dot(W, X)
    Fs = np.sqrt(np.multiply(F, F) + 1e-8)
    NFs, L2Fs = self.l2row(Fs)
    Fhat, L2Fn = self.l2row(NFs.T)


  def callback(self, W):
    """
    """
    sys.stdout.write("Iteration | %d\r" % (self.iteration))
    sys.stdout.flush()
    self.iteration += 1

  def visualiseLearnedFeatures(self):
    """
    """
    import matplotlib.pyplot as plt
    W = np.reshape(self.trained_W, (self.k, self.n_features), order="F")
    # each row of W is a learned feature
    extent = np.sqrt(self.n_features/self.channels)
    fig = plt.figure(facecolor="w")
    plt.ion()
    plotDims = int(np.ceil(np.sqrt(self.k)))
    for i in range(1,self.k+1):
      image = np.zeros((extent,extent,self.channels),dtype="float")
      ax = fig.add_subplot(plotDims, plotDims, i)
      for j in range(1,self.channels+1):
        image[:,:,j-1] += \
        np.reshape(W[i-1,(j-1)*extent*extent:j*extent*extent], \
        (extent, extent), order="F")
        image[:,:,j-1] = image[:,:,j-1]/np.max(image[:,:,j-1])
      image = image + 1
      image = image / 2.0
      cmap = "jet"
      if self.channels == 1:
        image = image[:,:,0]
        cmap = "hot"
      ax.imshow(image, interpolation="nearest", cmap=cmap)
      plt.axis("off")
    plt.ioff()
    plt.show()

  def save(self, out_file):
    """
    """
    output = open(out_file, "w")
    sio.savemat(output, {"k":int(self.k),
                         "channels":int(self.channels),
                         "n_features":int(self.n_features),
                         "max_iter":int(self.max_iter),
                         "trained_W": self.trained_W})

  def load(self, in_file):
    setup = sio.loadmat(save_file)
    self.k = int(setup["k"])
    self.channels = int(setup["channels"])
    self.max_iter = int(setup["max_iter"])
    self.n_features = int(setup["n_features"])
    self.trained_W = setup["trained_W"]

def computeNumericalGradient(func, params, *args):
  """
    Calculate the numerical apporximation to function gradients
  """
  
  data = args[0]
  numgrad = np.zeros(np.shape(params))
  perturb = np.zeros(np.shape(params))
  e = 0.0001
  for i in range(len(params)):
    # set perturbation vector
    perturb[i] = e
    loss1 = func((params - perturb), data)
    loss2 = func((params + perturb), data)
    # Compute Numerical Gradient
    numgrad[i] = (loss2 - loss1) / (2.0*e)
    perturb[i] = 0
  return numgrad

def checkGradients():
  """
  """
  def costFunction(W, *args):
    def l2row(X):
      n, m = np.shape(X)
      N = np.sqrt(np.sum(np.multiply(X, X), axis=1) + 1e-8)
      N_stack = np.tile(N, (m, 1)).T
      Y = np.divide(X, N_stack)
      return Y, N
    
    def l2rowg(X, Y, N, D):
      n, m = np.shape(X)
      N_stack = np.tile(N, (m, 1)).T
      firstTerm = np.divide(D, N_stack)
      sum = np.sum(np.multiply(D, X), 1)
      sum = sum / (np.multiply(N,N))
      sum_stack = np.tile(sum[np.newaxis], (np.shape(Y)[1],1)).T
      secondTerm = np.multiply(Y, sum_stack)
      return firstTerm - secondTerm
    
    X = args[0]
    n, m = np.shape(X)
    W = np.reshape(W, (k, n), order="F")
    # Feed forward
    F = np.dot(W, X)
    Fs = np.sqrt(np.multiply(F, F) + 1e-8)
    NFs, L2Fs = l2row(Fs)
    Fhat, L2Fn = l2row(NFs.T)
    # Compute objective function
    return np.sum(Fhat)

  k = 40
  n = 20
  # initialise
  W = np.random.rand(int(k),int(n))
  W = np.ravel(W, order="F")
  dataFile = "../data/naturalImages_patches_8x8.mat"
  data = sio.loadmat(dataFile)
  X = data["patches"][:n,:20]
  args = X, k
  
  sf = SparseFilter(k,1)
  cost, grad = sf.objective(X, W)
  numgrad = computeNumericalGradient(costFunction, W, *args)
  for i in range(len(numgrad)):
    print("%d\t%f\t%f" % (i, numgrad[i], grad[i]))
    
  print("The above two columns you get should be very similar.")
  print("(Left-Your Numerical Gradient, Right-Analytical Gradient)")
  print()
  print("If your backpropagation implementation is correct, then")
  print("the relative difference will be small (less than 1e-9). ")
  
  diff = numgrad-grad
  print(diff)

