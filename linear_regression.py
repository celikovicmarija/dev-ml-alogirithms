"""
Implementacija LinearRegression algoritma za regresiju. 
Studentski domaci zadatak, FON, Razvoj algoritama masinskog ucenja
Datum izmene Apr 4, 2022
"""
import pandas as pd
import numpy as np
import math

class LinearRegression:
    """
    Klasa Linearne regresije. Radi opcionu normalizaciju, kroz funkciju ucenja podesava tezine atributa,
    koriscenjem stochastic gradient descent metode- update-ovanjem parametara nakon svakog slucaja.
    Radi opciono i regularizaciju preko parametra lambda, i automatski uci parametar alfa preko alfa scheduling-a.
    
    PARAMETRI:
        model: numpy array, tj sa koeficijentima za svaki atribut
        x_mean: numpy array, sa prosecima za svaki atribut u datasetu. Cuva se radi kasnije transformacije
        x_std: numpy array, sa standardnim devijacijama u datasetu po svakom atributu. Cuva se radi kasnije transformacije.
    """
    
    def __init__(self):
        self.model=None
        self.x_mean=None
        self.x_std=None
        self.normalize=True
    
    def learn(self, data, target_attribute, epochs=50, alpha=0.02, lambda_=1, normalize=True):
        """
        Funkcija kojom se uci model linearne regersije. 
        

        Parameters
        ----------
        data : DataFrame
            Skup podataka na osnovu kojeg se kreira model linearne regresije.
        target_attribute : string
            Naziv atributa u skupu podataka koje je potrebno modelovati
        epochs : int, optional
            Broj iteracija, tj. epoha kroz koji se prolazi za otpimizaciju parametara. The default is 100.
        alpha : float, optional
            Learning rate, za racunanje prilagodjenih vrednosti modela. The default is 0.01.
        lambda_ : float, optional
            Jacina regularizacije koja se primenljuje pri racunanju funkcije greske. The default is 1.
            Sto je veci broj, jaca regularizacija (sprecavanje pretreniranja)
        normalize : Bool, optional
            Oznaka da li je potrebno normalizovati podatke. The default is True.

        Returns
        -------
        self
            Objekat klase LinearRegression.

        """
        y=data[target_attribute]
        X=data.drop(target_attribute, axis=1)
        self.normalize=normalize
        #opciona normalizacija
        if normalize==True:
            self.x_mean = X.mean()
            self.x_std = X.std()
            X = (X - self.x_mean) / self.x_std
            
        X['X0'] = 1
        m, n = X.shape 
        X = X.to_numpy() #da ne bude pandas objekat, vec numpy matrica
        y = y.to_numpy()
        print(X)
        w = np.random.random((1,n)) #random vrednosti za tezine za svaku kolonu, 0-1
        epochs_drop=30
        drop = 0.5
        gradient_global=np.ones((m,n))*np.inf
        print('iter, grad norm, mse, alpha')
        for epoch in range(epochs):
            for i in range(m): #prilagodjavanje vrednosti nakon svakog slucaja
                pred = X[i].dot(w.T) #matricno mnozenje
                err = pred - y[i]
                grad = err*X[i] / m +np.concatenate((2*lambda_*w[0][:-1],[0])) #regularizacija
                gradient_global[i]=np.array(grad)
                ##w[0]-because w is in the format [[]],[-1] because we are avoiding the penalisation of w0 (added as last column)
                #wo ne zavisi od atributa
                #and since we do not want to penalize it, we add zero at the end
                #zero must be like an array, else it will throw an error
                w-= alpha*grad #u smeru antigradijenta
            MSE = err.T.dot(err) / m #rezultat je jedan broj
            #cuvati globalni gradijent
            #grad_norm = abs(grad).sum()
            #print(grad_norm)
            print('Grad norm GLOBAL:')
            grad_norm_global = abs(gradient_global).mean(axis=0).sum()
            print(grad_norm_global)            
            #ako je grad_norm=0, stigao je u minimum
            #da zapravo stigne u mimimum!
            if grad_norm_global < 0.1:
                self.model=w.flatten()
                return self
            
            print(epoch, grad_norm_global, MSE, alpha)
            if epoch %epochs_drop==0:
                alpha *=math.pow(drop, math.floor((1+epoch)/epochs_drop))
            #alpha scheduling, nesto izmedju nekoliko metoda
            #inspiracija https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
                
                
        self.model=w.flatten() #stavljamo ga da bude u jednoj dimenziji
        return self
        
    
    def predict(self, data):
        """
        Function that predicts a continuos value based on the model learned.

        Parameters
        ----------
        data : DataFrame
            Input data for which the prediction is made.

        Returns
        -------
        predictions: pd.Series
            Pandas series with predictions for each case in the provided data

        """
        if self.normalize==True:
            data = (data-self.x_mean)/self.x_std
        data['X0']=1
        predictions=pd.Series(data.dot(self.model.T))
        return predictions
        




	
