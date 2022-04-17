"""
Implementacija AdaBoost algoritma za klasifikaciju. 
Studentski domaci zadatak, FON, Razvoj algoritama masinskog ucenja
Datum izmene Apr 3, 2022
"""

import numpy as np
import pandas as pd
import math
from random import choice

class AdaBoost:
    """
    AdaBoost classifier klasa. Pri instanciranju modela, inicijalizuju se osnovne promenljive.
    Model uci, tj. kreira ansambl base estimatora na osnovu podataka i zadatih parametara. 
    Nakon toga, model predvidja, prikazuje tacnosti predvidjanja.
    Moguce je definisati koliko ce se razlikovati sukcesivni modeli, tj.
    koliko ce se prilagodjavati greskama prethodnih modela.
    Moze da radi sa vise baznih algoritama, tj. omoguceno je da se pri svakoj iteraciji
    na slucajan nacin odabere koji ce se model kreirati.
    
    PARAMETRI
    
    base_model: object, default=None
        Osnovni weak learner koji kreira je sastavni deo ansambl algoritma. 
        Moze biti bilo koji klasifikator koji sadrzi metode fit, predict i atribut sample weights.
        
    ensemble_size: int, default=30
        Celobrojna vrednost koja oznacava velicinu ansambla, tj. broj njegovih base modela.
        t=1... T(ensemble_size)
        
    learning_rate: float, default=1.0
        Decimalni broj koji oznacava stepen prilagodjavanja modela greskama prethodnog modela.
        Veci broj oznacava vecu vrednost pojedinacnog klasifikatora u ansamblu
        
    base_learner_list: list, default:[]
        Ukoliko je zadata, u svakoj iteraciji se na nasumican nacin odabira jedan klasifikator 
        iz ove liste i dodaje se u ansambl. Ukoliko ne, uvek se koristi base_model.
        
    weights: numpy array, default=None
        Niz tezina koji oznacava znacaj pojedinacnih modela u ansamblu.
        Duzina zavisi od velicine ansabla.
        Ove tezine oznacavaju jacinu prilikom glasanja.
    """
    
    def __init__(self):
        self.base_model=None
        self.weights=None 
        self.ensemble=[] 
        self.learning_rate=1.0 
        self.base_learner_list=[] 

    
    def learn(self, X, y,  base_model, ensemble_size=30, learning_rate=1.0, base_learner_list=[]):
        """
        Uci model, tj. kreira odgovarajuci ansambl na osnovu ulaznih podataka i zadatih parametara
        PARAMETRI:
            X: ulazni podaci, DataFrame tipa, sa samo numerickim vrednostima
            y: target varijabla, Series tipa, sa vrednostima +1 i -1 za pozitivnu i negativnu klasu, respektivno
            ensemble_size: opciono, velicina ansambla
            learning_rate: opciono, stepen prilagodjavanja greskama prethodnog modela
            base_learner_list: opciono, lista iz koje se na slucajan nacin kreira model za i-ti clan ansambla
            
        VRACA:
            self: objekat AdaBoost klase
        
        """

        n = X.shape[0]
        alfas = pd.Series(np.array([1/n]*n), index=X.index) #znacaj svakog pojedinog slucaja
        print(f"Alfas: {alfas}, {type(alfas)}")
        self.weights = np.zeros(ensemble_size)   #tezine svakog pojedinacnog modela
        self.base_learner_list=base_learner_list 
        self.learning_rate=learning_rate
        self.base_model=base_model
        multiple=True if len(base_learner_list)>0 else False
        
        
        for t in range(ensemble_size):
            model = choice(self.base_learner_list).fit(X, y, sample_weight = alfas) if multiple else  base_model.fit(X, y, sample_weight = alfas)
            predictions = model.predict(X)
            error = (predictions!=y).astype(int)      # Series, greska i-tog modela u predvidjanju 1-greska, 0-pogodak, za racunanje WEt
            print(f"Error: {error}, {type(error)}")
            weighted_error = (error*alfas).sum()       
            # numpy.float64, WEt [0-1], sto manje to bolje, ukupna (otezana) greska sa tezinama instanci, za odredjivanje znacaja modela
            print(f"Weighted error: {weighted_error}, {type(weighted_error)}")
            w = self.learning_rate *1/2 * math.log((1-weighted_error)/weighted_error)   # float, Wt, preracunavanje tezina modela, acc/greska
            #sto je WEt manji, ln je veci, te je znacaj modela veci, i obrnuto.ako WEt=0.5-> Wt=0, mogu biti i negativne tezine
            print(f"W:\n{w}, {type(w)}")
            self.ensemble.append(model)
            self.weights[t] = w #znacaj modela

            factor = np.exp(-w*predictions*y) # za mnozenje sa starim alfa i dobijanje novih alfa tezina
            print(f"Factor:{factor} {type(factor)}")
            alfas = alfas * factor   # preracunavanje novih tezina instanci,  po formuli
        	
            z = alfas.sum()   # norma za normalizaciju, da bi suma na kraju bila nula.
            alfas = alfas/z
        print('Ensemble and weigths:')
        print(self.ensemble)
        print(self.weights)
        return self
    
    def predict(self, X, y):
        """
        Predvidja klase za ulazne podatke X
        Racuna se kao znak otezane sume vrednosti predikcija za svaki model u ansamblu
        
        PARAMETRI:
            X: ulazni podaci, Dataframe tipa, sa samo numerickim vrednostima
            
        VRACA:
            predictions: predikcije, Series tipa, za ulazne podatke X
        """
        predictions = pd.DataFrame([model.predict(X) for model in self.ensemble]).T
        #kolone su clanovi ansambla, redovi su slucajevi za koje se vrsi predikcija
        #dakle to su predikcij svakog za sebe
        print('Predictions:')
        print(predictions)
        print('Weighted predictions:\n')
        print(predictions.dot(self.weights))
        #sto je veci broj u apsolutnom smislu, to su se oni vise slozili
        predictions['ensemble'] = np.sign(predictions.dot(self.weights))
        #predictions['confidence']=predictions['ensemble']/np.sum(predictions.dot(self.weights).T)
        print(str(self.ensemble))
        print(predictions['ensemble'])
        print('Tacnost pojedinih modela')
        print((predictions.add(y, axis=0).abs()/2).mean()) # tacnost pojedinacnih modela i ansambla (komplementarnost)
        #accuraccy pojedinacnih modela, jer su oni po kolonama
        
        return predictions



