"""
Implementacija KMeans algoritma za klasterovanje. 
Studentski domaci zadatak, FON, Razvoj algoritama masinskog ucenja
Datum izmene Apr 3, 2022
"""
import pandas as pd
import numpy as np

class KMeans:
    """
    KMeans klasa. Pri instanciranju modela, inicijalizuju se osnovne promenljive.
    Prilikom ucenja, pronalazi se optimalan broj klastera primenom sihluette indeksa, kao i lokacija centroida.
    
    PARAMETRI:
        data_mean: float, prosek vrednosti atributa, cuva se radi kasnije transformacije
        data_std: float, standardna devijacija vrednosti atributa, cuva se radi kasnije transformacije
        normalize: bool, da li je potrebno odraditi normalizaciju
        attribute_weights: list, tezinski koeficijenti atributa
        distance: string, tip udaljenosti koji se koristi pri racunjanju
        centroids: dataFrame, sadrzi radne vrednosti centroida
        centroids_denormalized: dataFrame, sadrzi konacne vrednosti centroida u denormalizovanom obliku
        k: int, borj klastera koji je zadat
        num_iter: int, broj iteracija, tj. ponovnih inicijalizacija centroida, od kojih se kasnije bira najbolja
        total_ss: float, kvadratni zbir razlika svake tacke od globalnog proseka. Cuva se radi kasnije transformacije
        centroids_multiple_number_of_clusters: self.centroids_denormalized, cuva krajnje vrednosti centroida za razlicite brojeve klastera
        sihluette_score: float, cuva se radi odredjivanje broja klastera
        final_centroids: dataFrabe, cuva vrednosti atributa po svakom od klastera        
    """
    def __init__(self):
        self.data_mean = None
        self.data_std = None
        self.normalize = True
        self.attribute_weights = None
        self.distance = None
        self.centroids = {}
        self.centroids_denormalized = []
        self.k = None
        self.num_iter = None
        self.total_ss = None
        self.centroids_multiple_number_of_clusters = {}
        self.sihluette_scores = {}
        self.final_centroids = None
        self.p=None

    def calculate_distance(self, slucaj, b):
        """
        Racuna udaljenost dve tacke, u zavisnosti od tipa udaljenosti koji je zadat
        https://towardsdatascience.com/how-to-decide-the-perfect-distance-metric-for-your-machine-learning-model-2fa6e5810f11

        Parameters
        ----------
        slucaj : Series
            jedan slucaj iz pandas DataFrame-a za koji se racuna udaljenost.
        b : DataFrame
            Jedan red predstavlja jedan centroid.

        Returns
        -------
        distance: pandas Series
            Serija vrednosti koji predstavljaju udaljenosti tacke od centroida, po razlicitim metrikama.

        """

        if self.distance == 'eucledian':
            return (self.attribute_weights * (slucaj - b) ** 2).sum(axis=1)
        elif self.distance=='minkowski':
            return np.power((self.attribute_weights * abs(slucaj - b) ** self.p).sum(axis=1), 1/self.p)
        elif self.distance == 'manhattan':
            return (self.attribute_weights * (abs(slucaj - b))).sum(axis=1)
        elif self.distance == 'canberra':
            q=self.attribute_weights * (abs(slucaj - b)) / (abs(slucaj) + abs(b))
            return q.sum(axis=1)
        elif self.distance=='chebyshev':
                return abs(slucaj-b).max(axis=1)

    def calculate_distance_no_axis(self, slucaj, b):
        """
        Pomocna funkcija za racunanje udaljenosti kad ne postoji osa po kojoj se sabiraju vrednosti

        Parameters
        ----------
        slucaj : Series
            Jedan slucaj iz DatFrame-a.
        b : DataFrame
            Jedan slucaj iz DatFrame-a se odnosi na jedan centroid.

        Returns
        -------
        distance: float
            Serija vrednosti za udaljenosti slucaja od klastera po razlicitim kriterijumima
            Zna da radi sa kolonama s obzirom da ima nazive kolona

        """
        if self.distance == 'eucledian':
            return (self.attribute_weights * (slucaj - b) ** 2).sum()
        elif self.distance=='minkowski':
            return np.power((self.attribute_weights * abs(slucaj - b) ** self.p).sum(), 1/self.p)
        elif self.distance == 'manhattan':
            return (self.attribute_weights * (abs(slucaj - b))).sum()
        elif self.distance == 'canberra':
            q=self.attribute_weights * (abs(slucaj - b)) / (abs(slucaj) + abs(b))
            return q.sum()
        elif self.distance=='chebyshev':
                return abs(slucaj-b).max()

    def initialize_centroids(self, data, k, use_pdf=True):
        """
        Inicijalizacija centroida. Pametna, tako da je prva tacka odabrana na slucajan nacin
        iz skupa podataka, a ostale se biraju da su najudaljenije od postojeceg(postojecih) centroida
        
        Parameters
        ----------
        data : DataFrame
            Podaci iz kojih i za koje se racunaju centroidi.
        k : int
            Broj centroida koje je potrebno inicijalizovati.
        use_pdf : Bool, optional
            Da li je potrebno koristiti verovatnoce. The default is True.
            Ukoliko je to slucaj, najvecu verovatnocu odabira sledeceg centroida
            ima tacka koja je najudaljenija od prethodno odabranih centroida.
            Ukoliko je vrednost False, bira se najudaljenija tacka.

        Returns
        -------
        None.

        """
        n, m = data.shape
        initial_index = np.random.choice(range(data.shape[0]), )  # random prvi centroid
        centroids=data.iloc[initial_index, :].to_frame().T.reset_index(drop=True)
        for cl in range(k - 1):
            distances = []
            for point in range(n):  # za sve tacke udaljenost od centorida
                slucaj = data.iloc[point]
                # udaljenost od do najblizeg klastera, treba njemu najblizi da bude najudaljeniji
                distances.append(np.min(self.calculate_distance(slucaj, centroids)))
                # ovde se koristi euklidsko odstojanje, da li je u redu da tako ostane?

            if use_pdf:
                prob = distances / np.sum(distances)  
                # u teoriji racunas tako da najvecu verovatnocu izbora ima ona tacka koja je najudaljenija od svih centroida
                # false da tacka ne bi mogla biti izabrana vise puta
                centroids=centroids.append(data.iloc[np.random.choice(n, replace=False, p=prob), :]).reset_index(drop=True)
            else:
                ind = np.argmax(distances)  # ako ne koristi pdf, onda samo biras najudaljeniji centroid
                centroids=centroids.append(data.iloc[ind, :]).reset_index(drop=True)
                

        self.centroids[k] = centroids

    def learn(self, data, attribute_weights, distance, k_min=None, k_max=None, k=3, num_iter=10, normalize=True, p=None):
        """
        Parameters
        ----------
        data : DataFrame
            Podaci na osnovu kojih se kreiraju klasteri.
        attribute_weights : nparray
            Odredjuje vrednosti atributa u skupu podataka.
        distance : string
            Tip distance. Moguce vrednosti: 'eucledian','manhattan','canberra'
        k_min : int, optional
            Minimalan broj klastera koji se ispituje za pronalazenje optimalnog broja klastera.
        k_max : int, optional
            Maksimalan broj klastera koji se ispituje za pronalazenje optimalnog broja klastera..
        k : int, optional
            Fiksan broj klastera koji se trazi. The default is 3.
        num_iter : int, optional
            Broj inicijalizacija centroida, od kojih se posle bira najuspesnija. The default is 10.
        normalize : Bool, optional
            Da li je potrebno normalizovati podatke. The default is True.

        Returns
        -------
        self: object

        """
        n, m = data.shape
        self.attribute_weights = attribute_weights
        self.distance = distance
        self.normalize = normalize
        self.num_iter = num_iter
        self.p=p

        if normalize == True:
            self.data_mean = data.mean()
            self.data_std = data.std()
            data = (data - self.data_mean) / self.data_std
        
        sihluette_score_needed=True
        if k_min==None and k_max==None: 
            k_min=k
            k_max=k+1
            sihluette_score_needed=False
            
        for current_k in list(range(k_min, k_max)):
            print('Checking for '+str(current_k)+ ' clusters')
            qualities_all_iterations = {}
            centroids_all_iterations = {}
            for iter in range(num_iter):
                print(f"num_iter {iter}")
                self.initialize_centroids(data, current_k)
                old_quality = float('inf') #sto manji kvalitet, to je bolje, to su neke udaljenosti
                assign = np.zeros((n, 1))
                total_quality = 0
                for iteration in range(50):

                    quality = np.zeros(current_k)

                    for i in range(n):  # za sve tacke # 1. dodela tacaka klasterima
                        slucaj = data.iloc[i]
                        dist = self.calculate_distance(slucaj, self.centroids[current_k])  
                        # suma po redovima, suma udaljenosti tacaka od svih centroida
                        assign[i] = np.argmin(dist)  
                        # assign kom klasteru pripada svaka tacka array([[0.],[2.],[1.],[2.]]) #pozicija na kojoj se nalazi minimum niza

                    for c in range(current_k):  # po svim klasterima 	# 2. preracunavanje centroida
                        subset = data[assign == c]  # true, false... vraca sve slucajeve po datom klasteru.
                        assign=assign.flatten()
                        self.centroids[current_k].iloc[c,:] = subset.mean()  # preracunali smo ovaj centroid
                        quality[c] = subset.var().sum() * len(subset) #simple array
                    # da znas kad da stanes. varijanse po atributima za taj klaster, pa suma jer nas zanima ukupna
                    # varijansa je PROSECNO kvadratno odstupanje, a tebe zanima UKUPNO, zato mnozis sa len(subset)
                    #varijansa ovde je odstupanje od centroida po atributima, i kad se ne promeni u dve iteracije, staje se
                    total_quality = sum(quality)
                    #total_quality udaljenost svih tacaka od svojih centroida, sto manje to bolje
                    #quality racunas za svaku iteraciju (od 50) i poredis sa prethodnom
                    #iz iteracije u iteraciju mora da bude sve bolji rezultat
                    print(iteration, total_quality)
                    #dodat jos jedan uslov, ukoliko su slicne vrednosti u iteracijama, da prestane
                    if old_quality == total_quality or (round(total_quality/old_quality,3)>=0.999): break
                    old_quality = total_quality  # poredis sa prethodnim kvalitetom, ako se ne menja, kraj, ako ne, azuriraj old_quality

                qualities_all_iterations[iter] = old_quality
                centroids_all_iterations[iter] = self.centroids[current_k]

            self.centroids[current_k] = centroids_all_iterations[
                min(qualities_all_iterations, key=qualities_all_iterations.get)]
            self.centroids_multiple_number_of_clusters[current_k] = self.centroids[current_k]
            print(f"Best iteration number {min(qualities_all_iterations, key=qualities_all_iterations.get)}")

            print('Assigning elements to their clusters')
            assign = np.zeros((n, 1))
            # for the number of clusters current_k, add each point to a certain cluster
            for i in range(n):
                dist = self.calculate_distance(data.iloc[i], self.centroids_multiple_number_of_clusters[current_k])
                assign[i] = np.argmin(dist)
            # calculate intra-cluster distance
            # calculate sihluette index for each cluster
            # mean intra-cluster distance a
            # mean nearest-cluster distance that the point is not the part of b
            # (b - a) / max(a, b)
            print('Starting the calculation of sihluette score')
            self.sihluette_scores[current_k] = np.mean(self.calculate_sihluette_score_per_sample(data, assign[:, 0]))
            print(
                f"Sihluette score for iteration where number of clusters is {current_k}: {self.sihluette_scores[current_k]}")

        if sihluette_score_needed:
            print('Trying to find the optimal number of clusters...')
            self.final_centroids = self.centroids_multiple_number_of_clusters[
                max(self.sihluette_scores, key=self.sihluette_scores.get)]
            print(
                f"Centroids for the best sihluette score ({np.max(self.sihluette_scores)}), index {max(self.sihluette_scores, key=self.sihluette_scores.get)}")
 
            self.k = max(self.sihluette_scores, key=self.sihluette_scores.get)
        else:
            self.final_centroids=self.centroids[k]
            self.k=k
            
        destandardize = lambda  a: a * self.data_std + self.data_mean
        self.centroids_denormalized = self.final_centroids.apply(destandardize, axis=1)  
        return self

    def calculate_sihluette_score_per_sample(self, data, assign):
        """
        Funkcija koja za svaki slucaj izracunava sihluette score.

        Parameters
        ----------
        data : DataFrame
            Skup podataka za koji se vrednost racuna.
        assign : list
            Lista pripadnosti slucajeva odredjenom klasteru.

        Returns
        -------
        sihluette_score: float
            Moguce vrednosti su izmedju -1 i 1

        """
        n = len(assign)
        a = np.array([self.calculate_intra_cluster_distance(data, assign, i) for i in range(n)])
        b = np.array([self.nearest_cluster_distance(data, assign, i) for i in range(n)])
        return (b - a) / np.maximum(a, b)

    def calculate_intra_cluster_distance(self, data, assign, i):
        """
        Pomocna funkcija za racunanje sihluette scora. Racuna udaljenost 
        svih tacaka unutar jednog klastera.

        Parameters
        ----------
        data : DataFrame
            Skup podataka za koji se vrednost racuna.
        assign : list
            Lista pripadnosti slucajeva odredjenom klasteru.

        i : int
            Indeks klastera za koji se racuna intra cluster distance.

        Returns
        -------
        distance: float
            Prosecna medjusobna udaljenost elemenata unutar klastera.

        """
        indices = np.where(assign == assign[i])[0]
        return np.mean([self.calculate_distance_no_axis(data.iloc[i], data.iloc[j]) for j in indices if not i == j])

    def nearest_cluster_distance(self, data, assign, i):
        """
        Pomocna funkcija za racunanje sihluette scora.
        Vraca udaljenost od najblizeg klastera kome tacka ne pripada

        Parameters
        ----------
        data : DataFrame
            Skup podataka za koji se vrednost racuna.
        assign : list
            Lista pripadnosti slucajeva odredjenom klasteru.
        i : int
            Konkretan slucaj za koji se racuna udaljenost od najblizeg klastera.

        Returns
        -------
            Udaljenost od najblizeg klastera kome slucaj ne pripada

        """
        label = assign[i]
        return np.min([
            np.mean(
                [self.calculate_distance_no_axis(data.iloc[i], data.iloc[j]) for j in
                 np.where(assign == current_label)[0]]
            ) for current_label in set(assign) if not current_label == label
        ])

    def transform(self, data, level_to_warn=2000, separation_level=0.1):
        """
        Funkcija za pripisivanje klastera pojedinacnim slucajevima.
        Po potrebi se podaci normalizuju, i proveravaju se uslovi za upozorenje korisnika.

        Parameters
        ----------
        data : DataFrame
            Skup podataka za koji se racuna pripadnost postojecim klasterima.
        level_to_warn : float, optional
            Nivo za upozorenje korisnika ukoliko tacke nisu u najboljim klasterima. The default is 0.2.
        separation_level : float, optional
            Nivo za upozorenje korisnika kada klasteri nisu dobro medjusobno razdvojeni. The default is 0.1.

        Returns
        -------
        assign : list
            Vraca listu sa pripadnostima pojedinacnih slucajeva postojecim klasterima.

        """

        if self.normalize:
            data = (data - self.data_mean) / self.data_std

        n, m = data.shape
        assign = np.zeros((n, 1))
        #total_ss = 0
        for i in range(n):
            dist = self.calculate_distance(data.iloc[i], self.final_centroids)
            assign[i] = np.argmin(dist)
            #total_ss += ((data.iloc[i] - self.data_mean) ** 2).sum()
        
        assign=assign.flatten()
        quality=np.zeros(self.k)
        for c in range(self.k):
            subset=data[assign==c]
            quality[c]=subset.var().sum()*len(subset)
  
        quality=pd.Series(quality)
        print(quality)
        [print(f"Warning! Variance for cluster  {index} is {value} and is bigger than {level_to_warn}, therefore samples may not fit their assigned clusters well") for  index, value in quality.items() if value>level_to_warn ]

        """
        self.total_ss = total_ss
        ratio = self.calculate_ss_ratio(data)
        if ratio > level_to_warn:
            print(
                f"Warning! within_ss/total_ss ratio is {ratio}, therefore samples may not fit their assigned clusters well")
        """


        distances = self.calculate_cluster_separation()
        print(distances)
        if any(distances < separation_level):
            print(f"Warning! Cluster separation for at least one cluster is smaller than advised ({separation_level})")
        return assign

    def calculate_cluster_separation(self):
        """
        Funkcija koja racuna u kojoj meri su klasteri medjusobno razdvojeni.
        Kreira matricu medjusobne udaljenosti centroida.

        """
        distances = np.zeros((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                distances[i, j] = ((self.final_centroids.iloc[i] - self.final_centroids.iloc[j]) ** 2).sum()
        print('Cluster separation matrix: ')
        print(pd.DataFrame(distances))
        return distances.sum(1) / (distances.sum(1).sum())

    # totss (total_SS) - the sum of squared differences of each data point to the global sample mean
    # betweenss (between_SS) - the sum of squared differences of each cluster center to
    #   the global sample mean; the squared difference of each cluster center to
    #   the global sample mean is multiplied by the number of data points in that cluster
    # between_SS / total_SS - indicates how well the sample splits into clusters;
    #   the higher the ratio, the better clustering

    def calculate_ss_ratio(self, data):
        """
        Funkcija koja racuna koliko slucajevi dobro pripadaju klasterima
        Sto je veci, to je klasterovanje bolje.
        between_ss: kvadratni zbir razlika centroida od globalnog proseka.
            Mnozi se brojem slucajeva koji postoje u tom klasteru.
        total_ss: kvadratni zbir razlika svake tacke od globalnog proseka.
        
        PARAMETRI:
            data: DataFrame cija se velicina uzima za racunanje between_ss vrednosti
    
        VRACA:
            between_SS / total_SS float zaokruzen na 3 decimale
        """
        between_ss = 0
        for i in range(self.k):
            between_ss += ((self.final_centroids.iloc[i] - self.data_mean) ** 2).sum()

        self.between_ss = between_ss * data.shape[0]
        return round(self.between_ss / self.total_ss, 3)


pd.options.display.max_columns = 15

data = pd.read_csv('data/boston.csv')
model = KMeans()
model.learn(data.iloc[100:200,:], pd.Series(np.ones((1, data.shape[1])).flatten(), index=data.columns), 'eucledian', k_min=3, k_max=6, k=3, p=3)
data["klaster"] = model.transform(data)
data.head()
