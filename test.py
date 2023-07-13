import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import StandardScaler
import numpy as np
import joblib

train_image_path = 'dataset/train_images.idx3-ubyte'
train_label_path = 'dataset/train_labels.idx1-ubyte'
test_image_path  = 'dataset/test_image.idx3-ubyte'
test_label_path  = 'dataset/test_label.idx1-ubyte'

def image_process(path_file):
    with open(path_file, 'rb') as file:
        data = np.fromfile(file, np.uint8, offset=16)
        data = data / 255
        return data.reshape(-1, 28, 28)

def label_process(path_file):
    with open(path_file, 'rb') as file:
        data = np.fromfile(file, np.uint8, offset=8)
        return data

X_train = image_process(train_image_path)
X_test  = image_process(test_image_path)
y_train = label_process(train_label_path)
y_test  = label_process(test_label_path)

# PARAMETRES 
k = 4
distance_type = 'manhattan'
weight_type = 'uniform'

# prend en petit partie du data

xpetit_train = X_train[ : 500]
ypetit_train = y_train[ : 500]

xpetit_test = X_test[ : 10]
ypetit_test = y_test[ : 10]

xpetit_train = np.reshape(xpetit_train, (xpetit_train.shape[0], -1))
train = np.reshape(ypetit_train, (ypetit_train.shape[0], -1))
xpetit_test = np.reshape(xpetit_test, (xpetit_test.shape[0], -1))
ypetit_test = np.reshape(ypetit_test, (ypetit_test.shape[0], -1))





class RNN:
    def __init__(self , hidden_layer_sizes, alpha, iterations , normalize = True) :
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.iterations = iterations
        self.normalize = normalize
        
    def initialization_weights(self):
        low = - 0.1
        high = 0.1 
        self.weights = []
        self.baiais  = []       
        # Layer 1  
        w = np.random.uniform(low, high, size = (self.hidden_layer_sizes[0], self.input_layer_size ))
        self.weights.append(w)
        b = np.random.uniform(low, high, size = (w.shape[0] , 1))  
        self.baiais.append(b)   
                
        for i in range((len(self.hidden_layer_sizes)) - 1 ) :
            w = np.random.uniform(low, high, size = (self.hidden_layer_sizes[i+1], self.hidden_layer_sizes[i]))
            self.weights.append(w)
            b = np.random.uniform(low, high, (w.shape[0] , 1))  
            self.baiais.append(b)   
             
        #  outout layer
        w = np.random.uniform(low, high, size = ( self.output_layer_size , self.hidden_layer_sizes[-1]) )
        b = np.random.uniform(low, high, (w.shape[0] , 1))   
        self.baiais.append(b)    
        self.weights.append(w)         
        return self.weights , self.baiais
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def activation( z):
        return RNN.sigmoid(z)
    
    def lossFunction(self, weights, baiais):
        m = self.y_train.shape[0]        
        a = RNN.forwardPropagation(self.X_train , weights, baiais)
        return - np.sum(  self.YY @ np.log(a[-1]) +  ( 1 - self.YY ) @ np.log( 1 - a[-1] ) ) / m 
     
    def forwardPropagation(X, weights, baiais):
        a_all_layer = []
        a = np.transpose(X)
        a_all_layer.append(a)
        
        for i in range(len(weights)): 
            z = weights[i] @ a + baiais[i]
            a = RNN.activation(z)   
            a_all_layer.append(a)
            
        return a_all_layer
            
    def backPropagation(self):
        weights , baises = self.initialization_weights()
        cost_optimum = self.lossFunction(weights , baises)
                
        for i in range(self.iterations):
            a = RNN.forwardPropagation(self.X_train , weights , baises)  # 4 ACTIVATIONS

            dalta_weights = []
            dalta_baises = []
            
            # dz output layer
            dz = a[-1] - self.YY.T    # 10 * 500 
            dw = dz @ a[-2].T ###########################################

            dalta_weights.append(dw)
            dalta_baises.append(dz) # db = dz
            
            for L  in range( len(self.hidden_layer_sizes) - 1 , -1 , -1  ):             
                dz = weights[ L + 1].T @ dz * a[L + 1] * ( 1 - a[L + 1])
                
                dw = dz @ a[L].T ##################
                
                dalta_weights = [ dw ] + dalta_weights
                dalta_baises = [ dz ] + dalta_baises            
            
            
            mean_dalta_weights = [np.mean(dalta_weight , axis = 0).reshape((-1,1)) for dalta_weight in dalta_weights]
            mean_dalta_baises = [np.mean(dalta_baises , axis = 1).reshape((-1,1))  for dalta_baises in dalta_baises]
            
            for k in range(len(weights)):
               
                weights[k] = weights[k] - self.alpha * mean_dalta_weights[k].T
                baises[k] = baises[k] - self.alpha * mean_dalta_baises[k]  

            cost = self.lossFunction(weights, baises)
            if cost < cost_optimum : 
                cost_optimum = cost
                self.weights = weights
                self.baiais = baises   
        print(cost_optimum)
        return self.weights , self.baiais
        
    def fit(self, X_train , y_train):
        # recuperer nombre du classe 
        self.classes = np.unique(y_train)
        self.nombre_classes = len(self.classes)
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.YY = np.zeros((y_train.shape[0] , self.nombre_classes ))
        
        for i in range(self.nombre_classes):
            self.YY[ : , i ] = ( self.y_train == self.classes[i] ).astype(int)
            
        # faire normalisation
        if self.normalize :
            scaler = StandardScaler()
            scaler.fit( self.X_train)
            self.X_train = scaler.transform(self.X_train)

        self.input_layer_size =  X_train.shape[1]
        self.output_layer_size = self.nombre_classes
        
        self.backPropagation()
        
        return -1
        
    def predict(self, X_test ):       
        proba = RNN.forwardPropagation(X_test , self.weights, self.baiais)[-1]
        max_proba_classe = np.argmax(proba , axis=0)
        predect = self.classes[ max_proba_classe]
        return predect.reshape((-1, 1))
    
model = RNN( (15 , ) , 0.01 , 1000)
model.fit(xpetit_train , ypetit_train)

ypred = model.predict(xpetit_test)
accuracy = accuracy_score(ypetit_test, ypred)
print("Accuracy : {:.2f}%".format(accuracy * 100))
