import numpy as np
from tqdm import tqdm

class solver_v2:

  def __init__(self,L, X, k, lambda_param, beta_param, alpha_param, gamma_param):
    self.X = X
    self.p = X.shape[0]
    self.k = k
    self.n = X.shape[1]
    self.L = L


    n = self.n
    k = self.k
    p = self.p
    L = self.L

    self.thresh = 1e-10 # The 0-level

    # Basic initialization (Completely random)
    self.X_tilde = np.random.normal(0, 1, (k, n))
    
    self.C = np.random.normal(0,1,(p,k))
    self.C[self.C < self.thresh] = self.thresh
    
    self.w = np.random.normal(10, 1, (k*(k-1))//2)
    self.w[self.w < self.thresh] = self.thresh


    # Model Hyperparameters
    self.beta_param = beta_param
    self.alpha_param = alpha_param
    self.lambda_param = lambda_param
    self.gamma_param = gamma_param
    self.iters = 0
    self.lr0 = 1e-5

  def getLR(self):
    a = 0.99
    return self.lr0

  def calc_f(self):
    
    #w = self.w
    X_tilde = self.X_tilde
    beta_param = self.beta_param
    #Lw = self.L_operator(w)
    #L = np.load('L (5).npy')
    fw = 0

    fw += np.trace(X_tilde.T@self.C.T@self.L@self.C @X_tilde)
    print(" FGC  (XLX)")
    print(np.trace(X_tilde.T@self.C.T@self.L@self.C @X_tilde))
    # Added the tr(X.T L X) term
   # fw += ((beta_param*(np.linalg.norm(Lw)**2))/2)
    # Added the Frobbenius norm term
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    fw -= self.gamma_param*np.linalg.slogdet(self.C.T@self.L@self.C + J)[1]
    print(" FGC  (gamma)")
    print(self.gamma_param*np.linalg.slogdet(self.C.T@self.L@self.C + J)[1])
    # Added the log_det term
    fw += (self.alpha_param/2)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2
    print(" FGC  (alpha)")
    print((self.alpha_param/2)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2)
    # Added l2 norm || X - C*X_tilde ||
    fw += (self.lambda_param)/2*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2)
    print(" FGC  (lambda)")
    print((self.lambda_param)/2*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2))
    # Added L_1,2 norm || C ||
    return fw

  def update_X_tilde(self):
    #L = np.load('L (5).npy')
    L_tilde = self.C.T@self.L@self.C
    A = 2*L_tilde/(self.alpha_param)
    A = A + np.dot(self.C.T, self.C)
    b = np.dot(self.C.T, self.X)
    # Update 1
    self.X_tilde = np.linalg.pinv(A)@b

    # Update 2
    # lr = self.getLR()
    # self.X_tilde = self.X_tilde - lr*self.alpha_param*(A@self.X_tilde - b)

    # #new update:
    for i in range(len(self.X_tilde)):
      self.X_tilde[i] = (self.X_tilde[i]/(np.linalg.norm(self.X_tilde[i])))


    return None

  def grad_C(self):
    #L = np.load('L (5).npy')
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    v=np.linalg.pinv(self.C.T@self.L@self.C + J)
    gradC = np.zeros(self.C.shape)
    gradC += self.alpha_param*((self.C@self.X_tilde - self.X)@self.X_tilde.T)
    gradC += (self.lambda_param) * (np.abs(self.C) @ (np.ones((self.k, self.k))))
    gradC += -2*(self.gamma_param)*self.L@self.C@v
    gradC += 2*self.L@self.C@self.X_tilde@self.X_tilde.T
    
    return gradC

  def update_C(self, lr = None):
    if not lr:
      lr = 1/ (self.k)
    lr = self.getLR()
    C = self.C
    C = C - lr*self.grad_C()
    C[C<self.thresh] = self.thresh
    self.C = C
    C = self.C.copy()

    for i in range(len(C)):
      C[i] = C[i]/np.linalg.norm(C[i],1)

    self.C = C.copy()
    return None

  
  def fit(self, max_iters):
    ls = []
    MAX_ITER_INT = 100
    for i in tqdm(range(max_iters)):
      #for _ in range(MAX_ITER_INT):
        #self.update_w()
      for _ in range(MAX_ITER_INT):
        self.update_C(1/self.k)
      # for _ in range(MAX_ITER_INT):
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      #print(self.C@self.C.T)
      #print()

    return (self.C, self.X_tilde, ls )

  def New_fit(self):
    ls=[]
    MAX_ITER_INT = 100
    while(True):
      C_prev=self.C
      self.update_C(1/self.k)
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      if(np.linalg.norm(self.C-C_prev)<0.1): # we have set the threshold for stopping criteria as 0.1.
          return (self.C, self.X_tilde, ls )      
    return (self.C, self.X_tilde, ls )    

  def set_experiment(self, X, X_t):
    self.X = X
    self.X_tilde = X_t

class my_method:

  def __init__(self, L, X, k,U_k_vecs, lambda_param, beta_param, alpha_param, gamma_param,my_ev_param):
    self.X = X
    self.p = X.shape[0]
    self.k = k
    self.n = X.shape[1]
    self.U_k=U_k_vecs
    self.L = L

    n = self.n
    k = self.k
    p = self.p
    L = self.L

    
    self.thresh = 1e-10 # The 0-level

    # Basic initialization (Completely random)
    self.X_tilde = np.random.normal(0, 1, (k, n))
    
    #self.C = np.random.normal(0,1,(p,k))
    self.C = self.U_k
    self.C[self.C < self.thresh] = self.thresh
    
    self.w = np.random.normal(10, 1, (k*(k-1))//2)
    self.w[self.w < self.thresh] = self.thresh


    # Model Hyperparameters
    self.beta_param = beta_param
    self.alpha_param = alpha_param
    self.lambda_param = lambda_param
    self.gamma_param = gamma_param
    self.my_ev_param = my_ev_param
    self.iters = 0
    self.lr0 = 1e-5

  def getLR(self):
    a = 0.99
    return self.lr0

  def calc_f(self):
    
    #w = self.w
    X_tilde = self.X_tilde
    U_k=self.U_k
    beta_param = self.beta_param
    #Lw = self.L_operator(w)
    #L = np.load('L (5).npy')
    fw = 0

    fw -= self.my_ev_param*(np.trace(self.U_k@self.U_k.T@self.C@self.C.T))
    #fw += np.trace(X_tilde.T@self.C.T@L@self.C @X_tilde)
    #print(" \n my  (ev) \n")
    #print(np.trace(U_k@U_k.T@self.C@self.C.T))
    fw += (self.alpha_param/2)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2
    #print(" my  (alpha)")
    #print((self.alpha_param/2)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2)
    # Added l2 norm || X - C*X_tilde ||
    fw += (self.lambda_param)/2*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2)
    #print(" my  (lambda)")
    #print((self.lambda_param)/2*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2))
    # Added L_1,2 norm || C ||
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    fw -= self.gamma_param*np.linalg.slogdet(self.C.T@self.L@self.C + J)[1]

    return fw

  def update_X_tilde(self):
    #L = np.load('L (5).npy')
    L_tilde = self.C.T@self.L@self.C
    A = 2*L_tilde/(self.alpha_param)
    A = A + np.dot(self.C.T, self.C)
    b = np.dot(self.C.T, self.X)
    
    # Update 1
    self.X_tilde = np.linalg.pinv(A)@b

    # Update 2
    # lr = self.getLR()
    # self.X_tilde = self.X_tilde - lr*self.alpha_param*(A@self.X_tilde - b)

    # #new update:
    for i in range(len(self.X_tilde)):
      self.X_tilde[i] = (self.X_tilde[i]/(np.linalg.norm(self.X_tilde[i])))


    return None

  def grad_C(self):
    #L = np.load('L (5).npy')
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    v=np.linalg.pinv(self.C.T@self.L@self.C + J)
    gradC = np.zeros(self.C.shape)
    gradC += self.alpha_param*((self.C@self.X_tilde - self.X)@self.X_tilde.T)
    gradC += (self.lambda_param) * (np.abs(self.C) @ (np.ones((self.k, self.k))))
    gradC += -self.my_ev_param*(2*self.U_k@self.U_k.T@self.C)
    
    gradC += -2*(self.gamma_param)*self.L@self.C@v
    #gradC += 2*L@self.C@self.X_tilde@self.X_tilde.T
    
    return gradC

  def update_C(self, lr = None):
    if not lr:
      lr = 1/ (self.k)
    lr = self.getLR()
    C = self.C
    C = C - lr*self.grad_C()
    C[C<self.thresh] = self.thresh
    self.C = C
    C = self.C.copy()

    for i in range(len(C)):
      C[i] = C[i]/np.linalg.norm(C[i],1)

    self.C = C.copy()
    return None

  
  def fit(self, max_iters):
    ls = []
    MAX_ITER_INT = 100
    for i in tqdm(range(max_iters)):
      #for _ in range(MAX_ITER_INT):
        #self.update_w()
      for _ in range(MAX_ITER_INT):
        self.update_C(1/self.k)
      # for _ in range(MAX_ITER_INT):
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      #print(self.C@self.C.T)
      #print()

    return (self.C, self.X_tilde, ls )

  def New_fit(self):
    ls=[]
    MAX_ITER_INT = 100
    while(True):
      C_prev=self.C
      self.update_C(1/self.k)
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      if(np.linalg.norm(self.C-C_prev)<0.1): # we have set the threshold for stopping criteria as 0.1.
          return (self.C, self.X_tilde, ls )      
    return (self.C, self.X_tilde, ls )    

  def set_experiment(self, X, X_t):
    self.X = X
    self.X_tilde = X_t

class my_method_v2:

  def __init__(self, L, X, k,U_k_vecs, lambda_param, beta_param, alpha_param, gamma_param,my_ev_param):
    self.X = X
    self.p = X.shape[0]
    self.k = k
    self.n = X.shape[1]
    self.U_k=U_k_vecs
    self.L = L

    n = self.n
    k = self.k
    p = self.p
    L = self.L

    
    self.thresh = 1e-10 # The 0-level

    # Basic initialization (Completely random)
    self.X_tilde = np.random.normal(0, 1, (k, n))
    
    self.C = np.random.normal(0,1,(p,k))
    #self.C = self.U_k
    self.C[self.C < self.thresh] = self.thresh
    
    self.w = np.random.normal(10, 1, (k*(k-1))//2)
    self.w[self.w < self.thresh] = self.thresh


    # Model Hyperparameters
    self.beta_param = beta_param
    self.alpha_param = alpha_param
    self.lambda_param = lambda_param
    self.gamma_param = gamma_param
    self.my_ev_param = my_ev_param
    self.iters = 0
    self.lr0 = 1e-5

  def getLR(self):
    a = 0.99
    return self.lr0

  def calc_f(self):
    
    #w = self.w
    X_tilde = self.X_tilde
    U_k=self.U_k
    beta_param = self.beta_param
    #Lw = self.L_operator(w)
    #L = np.load('L (5).npy')
    fw = 0

    fw -= self.my_ev_param*(np.trace(U_k@U_k.T@self.C@self.C.T))
    #fw += np.trace(X_tilde.T@self.C.T@L@self.C @X_tilde)
    #print(" my  (ev)")
    #print(np.trace(U_k@U_k.T@self.C@self.C.T))
    fw += (self.alpha_param/2)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2
    #print(" my  (alpha)")
    #print((self.alpha_param/2)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2)
    # Added l2 norm || X - C*X_tilde ||
    fw += (self.lambda_param)/2*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2)
    #print(" my  (lambda)")
    #print((self.lambda_param)/2*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2))
    # Added L_1,2 norm || C ||
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    fw -= self.gamma_param*np.linalg.slogdet(self.C.T@self.L@self.C + J)[1]
	
        # Added l2 norm || X^T L X - X_c^T L_c X_c ||
    fw += 0.0001*np.linalg.norm(np.subtract(self.X.T@self.L@self.X, X_tilde.T@self.C.T@self.L@self.C @X_tilde))**2
    print(" my  new")
    print(0.0001*np.linalg.norm(np.subtract(self.X.T@self.L@self.X, X_tilde.T@self.C.T@self.L@self.C @X_tilde))**2)


    return fw

  def update_X_tilde(self):
    L=self.L
    C=self.C
    X_tilde=self.X_tilde
    X=self.X
    #L = np.load('L (5).npy')
    L_tilde = self.C.T@self.L@self.C
    A = 2*L_tilde/(self.alpha_param)
    A = A + np.dot(self.C.T, self.C)
    b = np.dot(self.C.T, self.X)
    
    # Update 1
    #self.X_tilde = np.linalg.pinv(A)@b

    # Update 2
    lr = self.getLR()
   
    T_0 = C@X_tilde
    #T_1 = (((X.T).dot(L)).dot(X) - (((T_0.T).dot(L)).dot(C)).dot(A))
    # functionValue = (np.linalg.norm(T_1, 'fro') ** 2)
    gradient = -(2*C.T@L@C@X_tilde@((L@X).T@X - (L@T_0).T@C@X_tilde) + 2*(L@C).T@C@X_tilde@(X.T@L@X-T_0.T@L@C@X_tilde))
    self.X_tilde = self.X_tilde - lr*(self.alpha_param*(A@self.X_tilde - b)+0.0001*gradient)

    # #new update:
    for i in range(len(self.X_tilde)):
      self.X_tilde[i] = (self.X_tilde[i]/(np.linalg.norm(self.X_tilde[i])))


    return None

  def grad_C(self):
    L=self.L
    C=self.C
    X_tilde=self.X_tilde
    X=self.X	
    #L = np.load('L (5).npy')
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    v=np.linalg.pinv(self.C.T@self.L@self.C + J)
    gradC = np.zeros(self.C.shape)
    gradC += self.alpha_param*((self.C@self.X_tilde - self.X)@self.X_tilde.T)
    gradC += (self.lambda_param) * (np.abs(self.C) @ (np.ones((self.k, self.k))))
    gradC += -10*(2*self.U_k@self.U_k.T@self.C)
    
    gradC += -2*(self.gamma_param)*self.L@self.C@v

    T_0=C@X_tilde
    gradC += -2*(L@C@X_tilde@((L@X).T@X-(L@T_0).T@C@X_tilde)@X_tilde.T+2*L.T@C@X_tilde@(X.T@L@X-T_0.T@L@C@X_tilde)@X_tilde.T)*0.0001

    #gradC += 2*L@self.C@self.X_tilde@self.X_tilde.T
    
    return gradC

  def update_C(self, lr = None):
    if not lr:
      lr = 1/ (self.k)
    lr = self.getLR()
    C = self.C
    C = C - lr*self.grad_C()
    C[C<self.thresh] = self.thresh
    self.C = C
    C = self.C.copy()

    for i in range(len(C)):
      C[i] = C[i]/np.linalg.norm(C[i],1)

    self.C = C.copy()
    return None

  
  def fit(self, max_iters):
    ls = []
    MAX_ITER_INT = 100
    for i in tqdm(range(max_iters)):
      #for _ in range(MAX_ITER_INT):
        #self.update_w()
      for _ in range(MAX_ITER_INT):
        self.update_C(1/self.k)
      # for _ in range(MAX_ITER_INT):
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      #print(self.C@self.C.T)
      #print()

    return (self.C, self.X_tilde, ls )

  def New_fit(self):
    ls=[]
    MAX_ITER_INT = 100
    while(True):
      C_prev=self.C
      self.update_C(1/self.k)
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      if(np.linalg.norm(self.C-C_prev)<0.1): # we have set the threshold for stopping criteria as 0.1.
          return (self.C, self.X_tilde, ls )      
    return (self.C, self.X_tilde, ls )    

  def set_experiment(self, X, X_t):
    self.X = X
    self.X_tilde = X_t

class my_method_exp:

  def __init__(self, L, X, k, U_k_vecs, a_1, a_2, a_3, a_4,a_5,a_6):
    self.X = X
    self.p = X.shape[0]
    self.k = k
    self.n = X.shape[1]
    self.U_k=U_k_vecs
    self.L = L

    n = self.n
    k = self.k
    p = self.p
    L = self.L

    
    self.thresh = 1e-10 # The 0-level

    # Basic initialization (Completely random)
    self.X_tilde = np.random.normal(0, 1, (k, n))
    
    self.C = np.random.normal(0,1,(p,k))
    #self.C = self.U_k
    self.C[self.C < self.thresh] = self.thresh
    
    self.w = np.random.normal(10, 1, (k*(k-1))//2)
    self.w[self.w < self.thresh] = self.thresh


    # Model Hyperparameters
    self.a_1 = a_1 # tr(X_c'L_cX_c)
    self.a_2 = a_2 # logdet(L_C + J)
    self.a_3 = a_3 # tr(U_kU_k'CC')
    self.a_4 = a_4 # || X'LX - X_c' L_c X_c||
    self.a_5 = a_5 # ||C||_1,2
    self.a_6 = a_6 # || X-CX_c||

    self.iters = 0
    self.lr0 = 1e-5

  def getLR(self):
    a = 0.99
    return self.lr0

  def calc_f(self):
    
    
    #w = self.w
    X_tilde = self.X_tilde
    U_k=self.U_k
    beta_param = self.beta_param
    #Lw = self.L_operator(w)
    #L = np.load('L (5).npy')
    fw = 0

    fw -= self.a_3*(np.trace(U_k@U_k.T@self.C@self.C.T))
    print(" my  (ev)")
    print(self.a_3*np.trace(U_k@U_k.T@self.C@self.C.T))

    fw += self.a_1*np.trace(X_tilde.T@self.C.T@L@self.C @X_tilde)
    print(" tr(X'LX)")
    print(self.a_1*np.trace(X_tilde.T@self.C.T@L@self.C @X_tilde))

    fw += (self.a_6)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2
    print(" || X-CX_c||")
    print((self.a_6)*(np.linalg.norm(np.subtract(self.X, np.dot(self.C, self.X_tilde))))**2)

    fw += (self.a_5)*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2)
    print(" ||C||_1,2")
    print((self.a_5)*((np.linalg.norm(np.dot(self.C, np.ones((self.k, 1)))))**2))
    
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    fw -= self.a_2*np.linalg.slogdet(self.C.T@self.L@self.C + J)[1]
    print(" logdet(L_c+J)")
    print(self.a_2*np.linalg.slogdet(self.C.T@self.L@self.C + J)[1])

        # Added l2 norm || X^T L X - X_c^T L_c X_c ||
    fw += self.a_4*np.linalg.norm(np.subtract(self.X.T@self.L@self.X, X_tilde.T@self.C.T@self.L@self.C @X_tilde))**2
    print(" || X^T L X - X_c^T L_c X_c ||")
    print(self.a_4*np.linalg.norm(np.subtract(self.X.T@self.L@self.X, X_tilde.T@self.C.T@self.L@self.C @X_tilde))**2)


    return fw

  def update_X_tilde(self):
    L=self.L
    C=self.C
    X_tilde=self.X_tilde
    X=self.X
    #L = np.load('L (5).npy')
    L_tilde = self.C.T@self.L@self.C
    A = 2*L_tilde/(self.alpha_param)
    A = A + np.dot(self.C.T, self.C)
    b = np.dot(self.C.T, self.X)
    
    # Update 1
    #self.X_tilde = np.linalg.pinv(A)@b

    # Update 2
    lr = self.getLR()
   
    T_0 = C@X_tilde
    #T_1 = (((X.T).dot(L)).dot(X) - (((T_0.T).dot(L)).dot(C)).dot(A))
    # functionValue = (np.linalg.norm(T_1, 'fro') ** 2)
    gradient = -(2*C.T@L@C@X_tilde@((L@X).T@X - (L@T_0).T@C@X_tilde) + 2*(L@C).T@C@X_tilde@(X.T@L@X-T_0.T@L@C@X_tilde))
    self.X_tilde = self.X_tilde - lr*(self.alpha_param*(A@self.X_tilde - b)+0.0001*gradient)

    # #new update:
    for i in range(len(self.X_tilde)):
      self.X_tilde[i] = (self.X_tilde[i]/(np.linalg.norm(self.X_tilde[i])))


    return None

  def grad_C(self):
    L=self.L
    C=self.C
    X_tilde=self.X_tilde
    X=self.X	
    #L = np.load('L (5).npy')
    J = np.outer(np.ones(self.k), np.ones(self.k))/self.k
    v=np.linalg.pinv(self.C.T@self.L@self.C + J)
    gradC = np.zeros(self.C.shape)
    gradC += self.alpha_param*((self.C@self.X_tilde - self.X)@self.X_tilde.T)
    gradC += (self.lambda_param) * (np.abs(self.C) @ (np.ones((self.k, self.k))))
    gradC += -10*(2*self.U_k@self.U_k.T@self.C)
    
    gradC += -2*(self.gamma_param)*self.L@self.C@v

    T_0=C@X_tilde
    gradC += -2*(L@C@X_tilde@((L@X).T@X-(L@T_0).T@C@X_tilde)@X_tilde.T+2*L.T@C@X_tilde@(X.T@L@X-T_0.T@L@C@X_tilde)@X_tilde.T)*0.0001

    #gradC += 2*L@self.C@self.X_tilde@self.X_tilde.T
    
    return gradC

  def update_C(self, lr = None):
    if not lr:
      lr = 1/ (self.k)
    lr = self.getLR()
    C = self.C
    C = C - lr*self.grad_C()
    C[C<self.thresh] = self.thresh
    self.C = C
    C = self.C.copy()

    for i in range(len(C)):
      C[i] = C[i]/np.linalg.norm(C[i],1)

    self.C = C.copy()
    return None

  
  def fit(self, max_iters):
    ls = []
    MAX_ITER_INT = 100
    for i in tqdm(range(max_iters)):
      #for _ in range(MAX_ITER_INT):
        #self.update_w()
      for _ in range(MAX_ITER_INT):
        self.update_C(1/self.k)
      # for _ in range(MAX_ITER_INT):
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      #print(self.C@self.C.T)
      #print()

    return (self.C, self.X_tilde, ls )

  def New_fit(self):
    ls=[]
    MAX_ITER_INT = 100
    while(True):
      C_prev=self.C
      self.update_C(1/self.k)
      self.update_X_tilde()
      ls.append(self.calc_f())
      self.iters+=1
      if(np.linalg.norm(self.C-C_prev)<0.1): # we have set the threshold for stopping criteria as 0.1.
          return (self.C, self.X_tilde, ls )      
    return (self.C, self.X_tilde, ls )    

  def set_experiment(self, X, X_t):
    self.X = X
    self.X_tilde = X_t