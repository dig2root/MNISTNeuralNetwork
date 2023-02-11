import numpy as np

# ----- Classe du réseau de neurones -----
class NeuralNetwork():

    def __init__(self):
        self.__iteration = 0

        # Dimension du réseau de neurones
        self.__nb_input_layer = 784
        self.__nb_hidden_layer = 16
        self.__nb_output_layer = 10

        # Initialisation des matrices des poids
        self.__hidden_layer_1_weights = 2*np.random.random((self.__nb_input_layer, self.__nb_hidden_layer)) -1
        self.__hidden_layer_2_weights = 2*np.random.random((self.__nb_hidden_layer, self.__nb_hidden_layer)) -1
        self.__output_layer_weights =  2*np.random.random((self.__nb_hidden_layer, self.__nb_output_layer)) -1


    def getOuputLayerWeights(self):
        return self.__output_layer_weights

    def __str__(self):
        return "Nombre d'itérations : " + str(self.__iteration)


    # ----------- Fonction d'activation sigmoïde -------------
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    # --------------------------------------------------------


    # --------- Dérivée de la fonction d'activation ----------
    def sigmoidPrime(self, x):
        return np.exp(-x)/((1 + np.exp(-x))**2)
    # --------------------------------------------------------
    
    
    def vectorToDigit(self, vector):
        chiffre = 0
        temp = 0
        for i, element in enumerate(vector[0]):
            if element > temp:
                temp = element
                chiffre = i
        return chiffre

    # --------- Méthode d'entier en format d'output ----------
    # Tâche : Convertit l'entier en entrée en un vecteur de taille (1x10)
    # correspondant à l'output attendu par le réseau de neurones
    # Paramètre(s) : - Entier entre 0 et 9
    # Sortie(s) : - Tableau numpy(1x10)
    def chiffreToVector(self, chiffre):
        result = np.zeros(10)
        result[chiffre] = 1
        return result
    # --------------------------------------------------------


    # ------- Méthode de conversion de format d'image --------
    # Tâche : Convertit l'image en un tableau numpy compatible
    # pour l'entrainement du réseau de neurones
    # Paramètre(s) : - Image(28x28) d'un chiffre manuscrit
    # Sortie(s) : - Tableau numpy(1x784)
    def imageToInput(self, image):
        normed = np.array(image / 255)
        result = normed.flatten()
        return result
    # --------------------------------------------------------


    # ----- Méthode d'entrainement du réseau de neurones -----
    # Tâche : Effectue une itération d'entrainement du réseau de neurones
    # en incrémentant un compteur à chaque itération pour tenir compte
    # de la longueur de l'entrainement.
    # Paramètre(s) : - Image(28x28) d'un chiffre manuscrit
    #                - Entier correspondant au chiffre manuscrit
    # Sortie(s) : vide
    def train(self, image, chiffre):
        self.__iteration += 1
        hidden_layer_1 = np.zeros(16)
        hidden_layer_2 = np.zeros(16)
        output_layer = np.zeros(10)

        # Feedforward
        input_layer = np.array([self.imageToInput(image)])
        hidden_layer_1 = self.sigmoid(hidden_layer_1 - np.dot(input_layer, self.__hidden_layer_1_weights)) # ajouter biai pour plus de précision ? 
        hidden_layer_2 = self.sigmoid(hidden_layer_2 - np.dot(hidden_layer_1, self.__hidden_layer_2_weights))
        output_layer = self.sigmoid(output_layer - np.dot(hidden_layer_2, self.__output_layer_weights))

        # Backpropagation
        output_layer_error = self.chiffreToVector(chiffre) - output_layer
        output_layer_delta = output_layer_error * self.sigmoidPrime(output_layer)

        hidden_layer_2_error = np.dot(output_layer_delta, self.__output_layer_weights.T)
        hidden_layer_2_delta = hidden_layer_2_error * self.sigmoidPrime(hidden_layer_2)

        hidden_layer_1_error = np.dot(hidden_layer_2_delta, self.__hidden_layer_2_weights.T)
        hidden_layer_1_delta = hidden_layer_1_error * self.sigmoidPrime(hidden_layer_1)

        # Modication des matrices de poids
        self.__hidden_layer_1_weights += np.dot(input_layer.T, hidden_layer_1_delta)
        self.__hidden_layer_2_weights += np.dot(hidden_layer_1.T, hidden_layer_2_delta)
        self.__output_layer_weights += np.dot(output_layer, output_layer_delta)

        #if (self.__iteration % 1000 == 0):
            #cout = str(np.mean(np.abs(output_layer_error)))
            #print("##### n = {} #####".format(self.__iteration))
            #print("Erreur : " + str(output_layer_error))
            #print("Cout : " + cout)
    # --------------------------------------------------------


    # --------------- Test du réseau de neurones -------------
    # Tâche : Applique le réseau de neurones à l'image en entrée et renvoi
    # l'entier correpondant.
    # Paramètre(s) : - Image(28x28) d'un chiffre manuscrit
    # Sortie(s) : - Entier correspondant au chiffre manuscrit
    def test(self, image):
        input_layer = np.array([self.imageToInput(image)])
        hidden_layer_1 = self.sigmoid(np.dot(input_layer, self.__hidden_layer_1_weights))
        hidden_layer_2 = self.sigmoid(np.dot(hidden_layer_1, self.__hidden_layer_2_weights))
        output_layer = self.sigmoid(np.dot(hidden_layer_2, self.__output_layer_weights))
        return self.vectorToDigit(output_layer)
    # --------------------------------------------------------