"""
Implementacija NaiveBayes algoritma za klasifikaciju. 
Studentski domaci zadatak, FON, Razvoj algoritama masinskog ucenja
Datum izmene Apr 4, 2022
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


class NaiveBayes:
    """
    NaiveBayes klasa.U learn metodi uci model, tj. postavlja odgovarajuce verovatnoce,
    dok u predikt nad novim skupom podataka vraca klase kao i verovatnoce pripadnosti klasama
    
    Podrazava rad sa numerickim podacima, umanjuje underflow efekat logovanjem verovatnoca, umanjuje overfitting 
    koriscenjem metode additive smoothing, gde je moguce zadati vrednost za jacinu tog atributa
    
    PARAMETRI:
        model: dict, za svaku vrednost nekog atributa sadrzi odgovarajuce verovatnoce
        class_probabilities: dict, verovatnoca da je u pitanju pojedina klasa za svaki slucaj
        numeric_cols: list, pomocni atribut radi transformacije numerickih u kategorickih promenljivih
    """

    def __init__(self):
        self.model = {}
        self.class_probabilities = {}
        self.numeric_cols = []

    def learn(self, data, class_attribute, alpha):
        """
        Metoda za ucenje Naivnog Bajesa. 
        Prvo odredjuje apriori, a zatim i uslovne verovatnoce, primenljuje additive smoothing.
        Ukoliko postoje numericke kolone, preko funkcije gustine uzima iz koje to normalne raspodele dolazi vrednost,
        tj. u recniku cuva prosek i standardnu devijaciju.
        
        PARAMETRI:
            data: DataFrame
                Skup podataka za koji se vrsi klasifikacija, tj. uci model
            class_attribute: string
                Naziv target kolone iz skupa podataka koji je osnov klasifikacije
            alpha: float
                Broj koji se koristi za additive smoothing radi sprecavanja overfittinga
                
        VRACA:
            self: objekat NaiveBayes klase
        """

        apriori = data[class_attribute].value_counts()
        print(apriori)
        #ADDITIVE SMOOTHING- stronger when you don't have a lot of data 
        #or you have a lot of noise- you don't trust your data
        #done for both apriori and conditional probabilities
        apriori = (apriori+ alpha)/(apriori.sum()+ (alpha*len(apriori)))
        #len(apriori)-NUMBER OF DIFFERENT VALUES FOR x, HOW MANY TIMES YOU ADDED alpha
        #https://towardsdatascience.com/introduction-to-naïve-bayes-classifier-fa59e3e24aaf
        #BEGGINING OF A SOLUTION OF PREVENTION OF UNDERFLOW PROBLEM
        #if log1p was not used, these probabilities would be negative
        self.model['_apriori'] = np.log(apriori) 

        self.numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        #selection of numeric columns
        for attribute in self.numeric_cols:
            self.model[attribute] = {'mean': data.groupby(class_attribute)[attribute].mean(), 'std': data.groupby(class_attribute)[attribute].std()}

        #selection of non-numeric columns
        for attribute in data.loc[:, ~data.columns.isin(self.numeric_cols)].drop(class_attribute, axis=1).columns:
            contingency_matrix = pd.crosstab(data[attribute], data[class_attribute]) #redovi su vrednosti atributa, kolone vrednosti target atributa
            contingency_matrix = (contingency_matrix+ alpha)/(contingency_matrix.sum(axis=0)+ (alpha*contingency_matrix.shape[0]))
            self.model[attribute] = np.log(contingency_matrix)
            #note: before and after log operation, the values do not sum up to one
        
        print('After learning, this is my model:')
        print(self.model)

        return self

    def predict(self, new_instance):
        """
        Funkcija ya odredjivanje pripadnosti nekoj klasi na osnovu naucenog modela.

        Parameters
        ----------
        new_instance : numpy array
            Instanca cija se klasna pripadnost odredjujeS

        Returns
        -------
        prediction : string
            Vrednost odgovarajuce klase.
        class_probabilites: dict
            verovatnoca pripadnosti svakoj od klasa

        """

        self.class_probabilities = {}
        
        for class_value in self.model['_apriori'].index:
            log_probability = 0
        

            for attribute in self.model:
                if attribute == '_apriori':
                    log_probability += self.model['_apriori'][class_value] #log odradjen ranije
                    
                elif attribute in self.numeric_cols:
                    log_probability += np.log(norm.pdf(new_instance[attribute],
                                                self.model[attribute]['mean'][class_value],
                                                self.model[attribute]['std'][class_value]))
                else:
                    log_probability += self.model[attribute][class_value][new_instance[attribute]] #log odradjen ranije
            self.class_probabilities[class_value] = np.exp(log_probability) #inverse of log1p is exp(a)-1, expm1(a)
       
        prediction = max(self.class_probabilities, key=self.class_probabilities.get)
        #kalkulacija stvarnih verovatnoca
        self.class_probabilities = {k: v / total for total in (sum(self.class_probabilities.values()),) for k, v in self.class_probabilities.items()}
       
        return prediction, self.class_probabilities