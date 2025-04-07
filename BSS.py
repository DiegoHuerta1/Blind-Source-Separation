# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:22:01 2024

@author: diego
"""


import scipy as sp
from scipy.optimize import minimize, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.linalg
from abc import ABC, abstractmethod

from Independence import Free_Independence, OV_Independence

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# para mezclar señales
class Mezcla_imagenes():
    '''
    Hace mezcla de señales usando una matriz A
    donde las señales son imagenes

    Se tiene una lista de s señales originales: x1, ..., xs
        donde x_i es una matriz n x n

    Se tiene una matriz A que es s x s

    Se hace la mezcla: y = Ax

    El resultado y es una lista de s señales mezcladas: y1, ..., ys
    '''

    # constructor
    def __init__(self, señales_x, matriz_A):
        '''
        señales_x es la lista de s matrices nxn
        matriz_A es una matriz sxs
        '''

        # poner como atrinutos
        self.señales_x = señales_x
        self.matriz_A = matriz_A

        # ver cuantas señales son
        self.s = len(señales_x)

        # ver que cada una es n x n
        self.n = señales_x[0].shape[0]
        for x_i in señales_x:
            assert x_i.shape == (self.n, self.n)

        # hacer un vector con las señales
        # dimensiones ns x n
        self.vector_x = np.vstack(señales_x)
        assert self.vector_x.shape == (self.n * self.s, self.n)

    # hacer la mezcla
    def mezclar_señales(self):
        '''
        Hacer y = Ax
        '''

        # hacer la mezcla: y= Ax
        # dimensiones: ns x n
        self.vector_y = np.kron(self.matriz_A,  np.eye(self.n)) @ self.vector_x

        # ademas del vector de dimensiones: ns x s
        # tener una lista de s matrices nxn
        self.señales_y = [self.get_matrices_componentes(self.vector_y, i) for i in range(self.s)]

        return self.señales_y

    # funcion auxiliar
    def get_matrices_componentes(self, vector_matrices, i):
        '''
        Dado un vector de matrices ns x n (s matrices n x n),
        tomar una matriz n x n componente  i (indice de 0 hasta s-1)
        '''
        matriz_componente = vector_matrices[i*self.n : (i+1)*self.n]
        return matriz_componente


    # ver las originales y la mezcla
    def ver_mezcla(self, figsize = None):

        # si no hay tamaño poner uno
        if figsize is None:
            figsize = (6, 3*self.s)

        # dos columnas, originales y mezcladas
        fig, ax = plt.subplots(self.s, 2, figsize = figsize)

        # ir llenando de
        for i in range(self.s):

            # poner la x_i
            ax[i, 0].imshow(self.señales_x[i], cmap='gray')
            ax[i, 0].axis('off')
            ax[i, 0].set_title(f'x_{i+1}')

            # poner la y_i
            ax[i, 1].imshow(self.señales_y[i], cmap='gray')
            ax[i, 1].axis('off')
            ax[i, 1].set_title(f'y_{i+1}')

        # finalizar
        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

## Clase abstracta de un metodo de separacion de señales
class Algoritmo_separacion_imagenes(ABC):
    '''
    Clase abstracta de un algoritmo de separacion de señales
    Aplicado al caso donde las señales son imagenes (matrices n x n)

    Se tienen s señales y = y1, ..., y_s
        donde y_i es una matriz n x n

    Se supone que y = Ax
    para una matriz desconocida A que es s x s
    y unas señales originales x = x1, ..., x_s

    Se intenta recuperar A y x haciendo ciertas suposiciones
    '''

    # constructor
    def __init__(self, señales_y):

        # poner las señales y como atributo
        self.señales_y = señales_y

        # ver cuantas señales son
        self.s = len(señales_y)

        # ver que cada una es n x n
        self.n = señales_y[0].shape[0]
        for y_i in señales_y:
            assert y_i.shape == (self.n, self.n)

        # hacer un vector con las señales
        # dimensiones ns x n
        self.vector_y = np.vstack(señales_y)
        assert self.vector_y.shape == (self.n * self.s, self.n)

    # ----------------------------------------------------------------------------------------

    # funciona: depnde de cada algoritmo
    @abstractmethod
    def funcional(self, Z):
        '''
        Funcional usado para calcular las covarianzas
        \bar{Y_i} = Y_i - funcional(Y_i) I_n
        C_{i,j} = funcional(\bar{Y_i}, \bar{Y_j}^T)
        '''
        pass

    # cumulante 4 evaluado en una matriz
    @abstractmethod
    def cumulante_4(self, Z):
        '''
        Cumulante 4, depende de la independencia
        Unicamente importa el cumulante_4 evaluado en una sola matriz nxn Z
        Es decir, c_4(Z) (que en realidad es c_4(Z, Z, Z, Z))
        '''
        pass


    # norma
    @abstractmethod
    def norma(self, x):
        '''
        Norma en el codominio del cumulante_4

        Si el cumulante es evaluado en los complejos, pues es valor absoluto
        si es cumulante es operated valued, es una norma en B

        se usa en el problema de optimizacion
        '''
        pass

    # ----------------------------------------------------------------------------------------

    # covariancas
    def compute_covarianzas(self):
        '''
        Calcular la matriz sxs de covarianzas
        '''

        # primero centrar todas las y
        señales_y_bar = [y_i - self.funcional(y_i) * np.eye(self.n) for y_i in self.señales_y]
        vector_y_bar = np.vstack(señales_y_bar)

        # inicar la matriz de covarianzas vacia
        self.C = np.zeros((self.s, self.s))

        # iterar en cada par i, j
        for i in range(self.s):
            for j in range(self.s):
                # poner la entrada C_{i,j}
                self.C[i, j] = self.funcional(señales_y_bar[i] @ señales_y_bar[j].T)

        # ya se tiene C, tambien calcular C^{1/2} y C^{-1/2}
        self.C_power_un_medio = sp.linalg.sqrtm(self.C)
        self.C_power_menos_un_medio = np.linalg.inv(self.C_power_un_medio)

        # obtener las señales despues de whitening
        self.vector_y_white = self.aumentar_matriz(self.C_power_menos_un_medio) @ vector_y_bar

        # devolver C
        return self.C

    # ----------------------------------------------------------------------

    # funcionaes auxiliares para la optimizacion

    def aumentar_matriz(self, matriz):
        '''
        Aumentar una matrz s x s para que sea ns x ns
        se hacen bloques que son la identidad por una constante
        '''
        # prouducto de kronecker con la identidad
        return np.kron(matriz,  np.eye(self.n))


    def get_matrices_componentes(self, vector_matrices, i):
        '''
        Dado un vector de matrices ns x n (s matrices n x n),
        tomar una matriz n x n componente  i (indice de 0 hasta s-1)
        '''
        matriz_componente = vector_matrices[i*self.n : (i+1)*self.n]
        return matriz_componente


    def cantidad_maximizar(self, W):
        '''
        La cantidad que debe de ser maximizada por la matriz W
        evaluada en una matriz especifica
        '''

        # primero calcular el producto de matrices
        # WCY = self.aumentar_matriz(W.T @ self.C_power_menos_un_medio) @ self.vector_y  # VERSION TESIS SAUL
        WCY = self.aumentar_matriz(W.T) @ self.vector_y_white # usar señales centradas

        # ir calculando cada termino de la suma
        cantidad_final = 0
        # i = 1 hasta s (bajar un indice)
        for i in range(self.s):
            # siguiendo la formula
            cantidad_final += self.norma( self.cumulante_4( self.get_matrices_componentes(WCY, i) ) )

        # devolvere toda la suma
        return cantidad_final


    def vector_to_matrix(self, vector):
        '''
        Pasar de un vector de s^2 elementos
        a una matriz sxs
        '''
        return vector.reshape(self.s, self.s)


    def matrix_to_vector(self, matrix):
        '''
        Pasar de una matriz sxs
        a un vector de s^2 elementos
        '''
        return matrix.flatten()


    def punto_inicial_W(self, iniciar_W_ortogonal):
        '''
        Devuelve un vector con las entradas de W aleatorio
        Usado como valor inicial para la optimizacion

        Se puede forzar a que sea ortogonal,
        usando una descomposicion QR
        '''

        # se quiere ortogoanl
        if iniciar_W_ortogonal:
            # tomar una aleatoria, y descomponer en QR, tomar Q
            Q, _ = np.linalg.qr(np.random.rand(self.s, self.s))
            vector_W_inicial = self.matrix_to_vector(Q)
        # solo aleatoria
        else:
            vector_W_inicial = np.random.rand(self.s**2)

        return vector_W_inicial


    # ----------------------------------------------------------------------

    # Problema de optimizacion
    def resolver_optimizacion(self, repeticiones_optimizacion, iniciar_W_ortogonal):
        '''
        Encuentra la matriz W_hat que maximiza la cantidad deseada
        Resuelve la optimizacion usando Byrd-Omojokun Trust-Region SQP
        '''

        # hacer la funcion a minimizar, toma un vector de s^2 elementos
        def funcion_objetivo_minimizar(vector_W):
            return -1 * self.cantidad_maximizar( self.vector_to_matrix(vector_W) )


        # calcular producto interno para pares de filas de W
        def restriccion_productos_internos(vector_W):
            W = self.vector_to_matrix(vector_W)
            # calcular para cada par de filas (i, j)
            restricciones = []
            for i in range(self.s):
                for j in range(i, self.s):
                    if i == j:
                        # el producto interno debe ser 1
                        restricciones.append(W[i] @ W[i] - 1)
                    else:
                        # el producto interno debe ser 0
                        restricciones.append(W[i] @ W[j])
            return np.array(restricciones)

        # crear la restriccion, esta funcion debe ser 0 en todas sus entradas
        constraint = NonlinearConstraint(restriccion_productos_internos, 0, 0)

        # se va a hacer la optimizacion varias veces, guardar
        self.best_valor_objetivo_opti = float('-inf')  # el mejor, considerando el problema de maximizacion
        self.valores_objetivo_opti = []

        # hacer las iteraciones
        for _ in range(repeticiones_optimizacion):

            # tomar el punto inicial para W
            vector_W_inicial = self.punto_inicial_W(iniciar_W_ortogonal)

            # hacer la minimizacion (mazimizacion)
            results_opti = minimize(funcion_objetivo_minimizar, vector_W_inicial,
                                    constraints= constraint, method = "trust-constr")

            #valor de la función objetivo
            valor_objetivo = -1 * results_opti.fun  # estamos maximizando

            # guardar todos
            self.valores_objetivo_opti.append(valor_objetivo)

            # Guardar si es el mejor resultado hasta ahora
            if valor_objetivo > self.best_valor_objetivo_opti:
                # poner como atributos
                self.best_valor_objetivo_opti = valor_objetivo
                self.best_valores_w_opti = results_opti.x
                self.best_results_opti = results_opti

        # fin de las repeticiones

        # obtener la matriz W_hat
        self.W_hat = self.vector_to_matrix( self.best_valores_w_opti )

        return self.W_hat



    # ----------------------------------------------------------------------
    # METODO PRINCIPAL

    def separar_señaes(self, repeticiones_optimizacion = 3, iniciar_W_ortogonal = True):
        '''
        Metodo principal
        Estima A^{-1} para obtener estimaciones x_hat de x

        Es decir, separa las señales y en x_hat,
        estimando las señales originales desconocidas x

        repeticiones_optimizacion - numero de veces que
        se intenta resolver el problema de optimizacion
        iniciar_W_ortogonal - indicar si el valor inicial para
        la optimizacion es una matriz ortogonal o no
        '''

        # calcular covarianzas
        self.compute_covarianzas()
        # hacer optimizacion para encontar W_hat
        self.resolver_optimizacion(repeticiones_optimizacion, iniciar_W_ortogonal)

        # calcular A_hat
        self.A_hat = self.C_power_un_medio @ self.W_hat
        # invertir
        self.A_hat_inversa = np.linalg.inv(self.A_hat)

        # calular x_hat en vector de matrices y en lista
        self.vector_x_hat = self.aumentar_matriz(self.A_hat_inversa) @ self.vector_y
        self.x_hat = [self.get_matrices_componentes(self.vector_x_hat, i) for i in range(self.s)]

        return self.x_hat

    # ----------------------------------------------------------------------
    # Visualizaciones

    # ver solo x_hat
    def ver_señales_rescatadas(self):
        '''
        Ver las señales x_hat
        '''

        # ver las s señales
        fig, ax = plt.subplots(self.s)

        # por cada i
        for i in range(self.s):
            # dibujar x_hat_i
            ax[i].imshow(self.x_hat[i], cmap='gray')
            ax[i].axis('off')
            ax[i].set_title(f'x_hat_{i+1}')

        # finalizar
        fig.suptitle("Señales rescatadas")
        plt.tight_layout()
        plt.show()

    def ver(self, señales_x_originales = None, figsize = None):
        '''
        Ver señales rescatadas x_hat, señales mezcladas y
        y opcionalmente tambien las señales originales
        '''

        # si no se tienen las originales
        if señales_x_originales is None:

            # solo poner y, x_hat

            # si no hay tamaño poner
            if figsize is None:
                figsize = (6, self.s*3)
            fig, ax = plt.subplots(self.s, 2, figsize = figsize)

            # ir llenando
            for i in range(self.s):
                # poner la y_i
                ax[i, 0].imshow(self.señales_y[i], cmap='gray')
                ax[i, 0].axis('off')
                ax[i, 0].set_title(f'y_{i+1}')
                # poner la x_hat_i
                ax[i, 1].imshow(self.x_hat[i], cmap='gray')
                ax[i, 1].axis('off')
                ax[i, 1].set_title(f'x_hat_{i+1}')

            # finalizar
            fig.suptitle("Señales mezcladas y rescatadas")
            plt.tight_layout()
            plt.show()

        # si es que se tienen las originales
        else:

            # poner x, y, x_hat

            # si no hay tamaño poner
            if figsize is None:
                figsize = (9, self.s*3)
            fig, ax = plt.subplots(self.s, 3, figsize = figsize)

            # ir llenando
            for i in range(self.s):
                # poner la x_i
                ax[i, 0].imshow(señales_x_originales[i], cmap='gray')
                ax[i, 0].axis('off')
                ax[i, 0].set_title(f'x_{i+1}')
                # poner la y_i
                ax[i, 1].imshow(self.señales_y[i], cmap='gray')
                ax[i, 1].axis('off')
                ax[i, 1].set_title(f'y_{i+1}')
                # poner la x_hat_i
                ax[i, 2].imshow(self.x_hat[i], cmap='gray')
                ax[i, 2].axis('off')
                ax[i, 2].set_title(f'x_hat_{i+1}')

            # finalizar
            fig.suptitle("Señales originales, mezcladas y rescatadas")
            plt.tight_layout()
            plt.show()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# implementar FCA
class FCA(Algoritmo_separacion_imagenes):
    '''
    Separacion de señales con entradas libres.

    Especificaciones:
        Funcional = 1/n Tr
        cumulante_4 = kappa_4 (cumulante libre)
        norma =  valor absoluto
    '''

    # implementar el funcional
    def funcional(self, Z):
        return (1/self.n) * Z.trace()

    # implementar el cumulante
    def cumulante_4(self, Z):
        return self.funcional( np.linalg.matrix_power(Z, 4) ) - 2 * ( self.funcional( np.linalg.matrix_power(Z, 2) ) )**2

    # implemetar la norma
    def norma(self, x):
        return np.abs(x)

# ---------------------------------------------------------------------------

# implementar BCA
class BCA(Algoritmo_separacion_imagenes):
    '''
    Separacion de señales con entradas libres en sentido booleano.

    Parametro:
        Vector unitario v de Rn

    Especificaciones:
        Funcional = v^T Z v
        cumulante_4 = b_4 (cumulante booleano)
        norma =  valor absoluto
    '''

    # sobre escribir el cosntructor, para delimitar el vector v
    def __init__(self, señales_y, vector_v):
        # llamar al constructor del padre
        super().__init__(señales_y)

        # asegurarnos que v es un vector unitario de Rn
        assert len(vector_v) == self.n
        assert np.isclose(1, np.dot(vector_v.T, vector_v) )

        # ponerlo como atributo
        self.vector_v = vector_v

    # implementar el funcional
    def funcional(self, Z):
        return self.vector_v.T @ Z @ self.vector_v

    # implementar el cumulante
    def cumulante_4(self, Z):
        return self.funcional(np.linalg.matrix_power(Z, 4)) - ( self.funcional(np.linalg.matrix_power(Z, 2)) )**2

    # implemetar la norma
    def norma(self, x):
        return np.abs(x)

# ---------------------------------------------------------------------------

# implementar OVFCA
class OVFCA(Algoritmo_separacion_imagenes):
    '''
    Separacion de señales con entradas libres con amalgacion sobre B,
    esto con respecto a una esperanza condicional
    F: A -> B
        A: matrices nxn
        B: subconjunto  (subalgebra) de matrices nxn


    Especificaciones:
        Funcional = 1/n Tr
        cumulante_4 = kappa_4^B (cumulante valuado en operadores )
        norma =  norma en B
    '''

    # sobre escribir el cosntructor, para delimitar
    # una esperanza condicional y una norma en su imagen
    def __init__(self, señales_y, esperanza_condicional_F, norma_en_B):
        # llamar al constructor del padre
        super().__init__(señales_y)

        # poner estos como atributos
        self.F = esperanza_condicional_F
        self.norma_en_B = norma_en_B


    # implementar el funcional
    def funcional(self, Z):
        return (1/self.n) * Z.trace()

    # implementar el cumulante
    def cumulante_4(self, Z):
        return self.F( np.linalg.matrix_power(Z, 4) ) - ( self.F(np.linalg.matrix_power(Z, 2)) )**2 - self.F( Z @ self.F(np.linalg.matrix_power(Z, 2)) @ Z)

    # implemetar la norma
    def norma(self, x):
        return self.norma_en_B(x)

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Metodos de separacion de señales con cumulantes no centrados

# implementar FCA no centrado
class FCA_no_centrado(Algoritmo_separacion_imagenes):
    '''
    Separacion de señales con entradas libres.

    Especificaciones:
        Funcional = 1/n Tr
        cumulante_4 = kappa_4 (cumulante libre) caso general, no asume variables centradas
        norma =  valor absoluto
    '''

    # implementar el funcional
    def funcional(self, Z):
        return (1/self.n) * Z.trace()

    # implementar el cumulante
    def cumulante_4(self, Z):
        # calcular un cumulante libre no centrado
        # usar la clase que ya tengo
        obj_cumulant = Free_Independence(Z, None, functional_phi = self.funcional, product = lambda x, y: x @ y)
        return obj_cumulant.evaluate_cumulant(("A", "A", "A", "A"))

    # implemetar la norma
    def norma(self, x):
        return np.abs(x)

# implementar OVFCA no centrado
class OVFCA_no_centrado(Algoritmo_separacion_imagenes):
    '''
    Separacion de señales con entradas libres con amalgacion sobre B,
    esto con respecto a una esperanza condicional
    F: A -> B
        A: matrices cuadradas nxn
        B: subconjunto  (subalgebra) de matrices nxn


    Especificaciones:
        Funcional = 1/n Tr
        cumulante_4 = kappa_4^B (cumulante valuado en operadores ) caso general, no asume variables centradas
        norma =  norma en B

    Esta implementacion solo funciona en el caso que:
    n = 2N
    Es decir, A = M_2(C)  (A son matrices 2x2 cuyos elementos son matrices NxN)
    Los elemntos de C (es decir, las matirces NxN) forman un espacio de probabilidad no conmutativo
    Este es el caso de: Proposition 13 pagina 242 de Free Probability and Random Matrices
    con d = 2 y C matrices NxN
    Se utiliza la esperanza condicional mencionada en esa proposicion
    '''

    # sobre escribir el cosntructor, para delimitar
    # una esperanza condicional y una norma en su imagen
    def __init__(self, señales_y):
        # llamar al constructor del padre
        super().__init__(señales_y)

        # obtener N
        assert self.n%2 == 0
        self.N = int(self.n/2)


    # implementar el funcional
    def funcional(self, Z):
        # assert Z.shape == (2 * self.N, 2 * self.N)
        return (1/self.n) * Z.trace()

    # delimitar el funcional phi que actua en las matrices componente
    # es decir, en una matrix NxN
    def funcional_phi(self, Z_bloque):
        # assert Z_bloque.shape == (self.N, self.N)
        return (1 / self.N) * Z_bloque.trace()

    # implementar el cumulante
    def cumulante_4(self, Z):
        # Z es una matriz a bloques, es una mariz 2x2 donde cada elemento es una matriz NxN
        # separar los elementos
        # assert Z.shape == (2 * self.N, 2 * self.N)
        Z11 = Z[0:self.N, 0:self.N]
        Z12 = Z[0:self.N, self.N:2 * self.N]
        Z21 = Z[self.N:2 * self.N, 0:self.N]
        Z22 = Z[self.N:2 * self.N, self.N:2 * self.N]
        # ponerlo en formato correcto
        Z_format = {(1, 1): Z11, (1, 2): Z12, (2, 1): Z21, (2, 2): Z22}
        # calcular un cumulante libre no centrado
        # usar la clase que ya tengo
        obj_cumulant = OV_Independence(Z_format, {(1,): 2}, d = 2,
                                       functional_phi=self.funcional_phi, product=lambda x, y: x @ y)
        cuarto_cumulante = obj_cumulant.evaluate_ov_cumulant(("A", "A", "A", "A"))
        # aumentarla para que sea 2N x 2N
        return np.kron(cuarto_cumulante, np.eye(self.N))


    # implemetar la norma
    def norma(self, x):
        # assert x.shape == (2*self.N, 2*self.N)
        return np.linalg.norm(x, 'fro')

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

class Algoritmo_separacion_imagenes_fast(ABC):
    '''
    Clase abstracta de un algoritmo de separacion de señales
    Aplicado al caso donde las señales son imagenes (matrices n x n)

    Se tienen s señales y = y1, ..., y_s
        donde y_i es una matriz n x n

    Se supone que y = Ax
    para una matriz desconocida A que es s x s
    y unas señales originales x = x1, ..., x_s

    Se intenta recuperar A y x haciendo ciertas suposiciones

    Intento hacer una implementacion mas eficiente
    '''

    # constructor
    def __init__(self, señales_y):

        # poner las señales y como atributo
        self.señales_y = señales_y

        # ver cuantas señales son
        self.s = len(señales_y)

        # ver que cada una es n x n
        self.n = señales_y[0].shape[0]
        for y_i in señales_y:
            assert y_i.shape == (self.n, self.n)


    # ----------------------------------------------------------------------------------------

    # funciona: depnde de cada algoritmo
    @abstractmethod
    def funcional(self, Z):
        '''
        Funcional usado para calcular las covarianzas
        \bar{Y_i} = Y_i - funcional(Y_i) I_n
        C_{i,j} = funcional(\bar{Y_i}, \bar{Y_j}^T)
        '''
        pass

    # cumulante 4 evaluado en una matriz
    @abstractmethod
    def cumulante_4(self, Z):
        '''
        Cumulante 4, depende de la independencia
        Unicamente importa el cumulante_4 evaluado en una sola matriz nxn Z
        Es decir, c_4(Z) (que en realidad es c_4(Z, Z, Z, Z))
        '''
        pass


    # norma
    @abstractmethod
    def norma(self, x):
        '''
        Norma en el codominio del cumulante_4

        Si el cumulante es evaluado en los complejos, pues es valor absoluto
        si es cumulante es operated valued, es una norma en B

        se usa en el problema de optimizacion
        '''
        pass

    # ----------------------------------------------------------------------------------------

    # covariancas
    def compute_covarianzas(self):
        '''
        Calcular la matriz sxs de covarianzas
        '''

        # primero centrar todas las y
        self.señales_y_bar = [y_i - self.funcional(y_i) * np.eye(self.n) for y_i in self.señales_y]

        # inicar la matriz de covarianzas vacia
        self.C = np.zeros((self.s, self.s))

        # iterar en cada par i, j
        for i in range(self.s):
            for j in range(self.s):
                # poner la entrada C_{i,j}
                self.C[i, j] = self.funcional(self.señales_y_bar[i] @ self.señales_y_bar[j].T)

        # ya se tiene C, tambien calcular C^{1/2} y C^{-1/2}
        self.C_power_un_medio = sp.linalg.sqrtm(self.C)
        self.C_power_menos_un_medio = np.linalg.inv(self.C_power_un_medio)

        # obtener las señales despues de whitening
        self.señales_y_white = self.multuplicar_matrix_con_vector_de_matrices(self.C_power_menos_un_medio, self.señales_y_bar)

        # devolver C
        return self.C

    # ----------------------------------------------------------------------

    # funcionaes auxiliares para la optimizacion

    def multuplicar_matrix_con_vector_de_matrices(self, M, z):
        """
        Dado una lista z de s matrices nxn (z pueden ser las señales y)
        y una matriz M que es sxs, calcular la multiplicacion,
        donde se ausme que z es un vector donde sus componentes son matrices.
        Es decir, se hace:

        np.kron(M, np.eye(n)) z

        y se deuevele el resutlado como una lista de matrices (de la forma de z)
        """

        # lista de matrices como resultado
        return [sum(M[i][j] * z[j] for j in range(self.s))
                for i in range(self.s)]



    def cantidad_maximizar(self, W):
        '''
        La cantidad que debe de ser maximizada por la matriz W
        evaluada en una matriz especifica
        '''

        # primero calcular el producto de matrices
        # WCY_list = self.multuplicar_matrix_con_vector_de_matrices(W.T @ self.C_power_menos_un_medio, self.señales_y) # VERSION TESIS SAUL
        WCY_list = self.multuplicar_matrix_con_vector_de_matrices(W.T, self.señales_y_white) # PRUEBA

        # ir calculando cada termino de la suma
        cantidad_final = 0
        # i = 1 hasta s (bajar un indice)
        for WCY_element in WCY_list:
            # siguiendo la formula
            cantidad_final += self.norma( self.cumulante_4( WCY_element ) )

        # devolvere toda la suma
        return cantidad_final


    def vector_to_matrix(self, vector):
        '''
        Pasar de un vector de s^2 elementos
        a una matriz sxs
        '''
        return vector.reshape(self.s, self.s)


    def matrix_to_vector(self, matrix):
        '''
        Pasar de una matriz sxs
        a un vector de s^2 elementos
        '''
        return matrix.flatten()


    def punto_inicial_W(self, iniciar_W_ortogonal):
        '''
        Devuelve un vector con las entradas de W aleatorio
        Usado como valor inicial para la optimizacion

        Se puede forzar a que sea ortogonal,
        usando una descomposicion QR
        '''

        # se quiere ortogoanl
        if iniciar_W_ortogonal:
            # tomar una aleatoria, y descomponer en QR, tomar Q
            Q, _ = np.linalg.qr(np.random.rand(self.s, self.s))
            vector_W_inicial = self.matrix_to_vector(Q)
        # solo aleatoria
        else:
            vector_W_inicial = np.random.rand(self.s**2)

        return vector_W_inicial


    # ----------------------------------------------------------------------

    # Problema de optimizacion
    def resolver_optimizacion(self, repeticiones_optimizacion, iniciar_W_ortogonal):
        '''
        Encuentra la matriz W_hat que maximiza la cantidad deseada
        Resuelve la optimizacion usando Byrd-Omojokun Trust-Region SQP
        '''

        # hacer la funcion a minimizar, toma un vector de s^2 elementos
        def funcion_objetivo_minimizar(vector_W):
            return -1 * self.cantidad_maximizar( self.vector_to_matrix(vector_W) )


        # calcular producto interno para pares de filas de W
        def restriccion_productos_internos(vector_W):
            W = self.vector_to_matrix(vector_W)
            # calcular para cada par de filas (i, j)
            restricciones = []
            for i in range(self.s):
                for j in range(i, self.s):
                    if i == j:
                        # el producto interno debe ser 1
                        restricciones.append(W[i] @ W[i] - 1)
                    else:
                        # el producto interno debe ser 0
                        restricciones.append(W[i] @ W[j])
            return np.array(restricciones)

        # crear la restriccion, esta funcion debe ser 0 en todas sus entradas
        constraint = NonlinearConstraint(restriccion_productos_internos, 0, 0)

        # se va a hacer la optimizacion varias veces, guardar
        self.best_valor_objetivo_opti = float('-inf')  # el mejor, considerando el problema de maximizacion
        self.valores_objetivo_opti = []

        # hacer las iteraciones
        for _ in range(repeticiones_optimizacion):

            # tomar el punto inicial para W
            vector_W_inicial = self.punto_inicial_W(iniciar_W_ortogonal)

            # hacer la minimizacion (mazimizacion)
            results_opti = minimize(funcion_objetivo_minimizar, vector_W_inicial,
                                    constraints= constraint, method = "trust-constr")

            #valor de la función objetivo
            valor_objetivo = -1 * results_opti.fun  # estamos maximizando

            # guardar todos
            self.valores_objetivo_opti.append(valor_objetivo)

            # Guardar si es el mejor resultado hasta ahora
            if valor_objetivo > self.best_valor_objetivo_opti:
                # poner como atributos
                self.best_valor_objetivo_opti = valor_objetivo
                self.best_valores_w_opti = results_opti.x
                self.best_results_opti = results_opti

        # fin de las repeticiones

        # obtener la matriz W_hat
        self.W_hat = self.vector_to_matrix( self.best_valores_w_opti )

        return self.W_hat



    # ----------------------------------------------------------------------
    # METODO PRINCIPAL

    def separar_señaes(self, repeticiones_optimizacion = 3, iniciar_W_ortogonal = True):
        '''
        Metodo principal
        Estima A^{-1} para obtener estimaciones x_hat de x

        Es decir, separa las señales y en x_hat,
        estimando las señales originales desconocidas x

        repeticiones_optimizacion - numero de veces que
        se intenta resolver el problema de optimizacion
        iniciar_W_ortogonal - indicar si el valor inicial para
        la optimizacion es una matriz ortogonal o no
        '''

        # calcular covarianzas
        self.compute_covarianzas()
        # hacer optimizacion para encontar W_hat
        self.resolver_optimizacion(repeticiones_optimizacion, iniciar_W_ortogonal)

        # calcular A_hat
        self.A_hat = self.C_power_un_medio @ self.W_hat
        # invertir
        self.A_hat_inversa = np.linalg.inv(self.A_hat)

        # calular x_hat
        self.x_hat = self.multuplicar_matrix_con_vector_de_matrices(self.A_hat_inversa, self.señales_y)

        return self.x_hat

    # ----------------------------------------------------------------------
    # Visualizaciones

    # ver solo x_hat
    def ver_señales_rescatadas(self):
        '''
        Ver las señales x_hat
        '''

        # ver las s señales
        fig, ax = plt.subplots(self.s)

        # por cada i
        for i in range(self.s):
            # dibujar x_hat_i
            ax[i].imshow(self.x_hat[i], cmap='gray')
            ax[i].axis('off')
            ax[i].set_title(f'x_hat_{i+1}')

        # finalizar
        fig.suptitle("Señales rescatadas")
        plt.tight_layout()
        plt.show()

    def ver(self, señales_x_originales = None, figsize = None):
        '''
        Ver señales rescatadas x_hat, señales mezcladas y
        y opcionalmente tambien las señales originales
        '''

        # si no se tienen las originales
        if señales_x_originales is None:

            # solo poner y, x_hat

            # si no hay tamaño poner
            if figsize is None:
                figsize = (6, self.s*3)
            fig, ax = plt.subplots(self.s, 2, figsize = figsize)

            # ir llenando
            for i in range(self.s):
                # poner la y_i
                ax[i, 0].imshow(self.señales_y[i], cmap='gray')
                ax[i, 0].axis('off')
                ax[i, 0].set_title(f'y_{i+1}')
                # poner la x_hat_i
                ax[i, 1].imshow(self.x_hat[i], cmap='gray')
                ax[i, 1].axis('off')
                ax[i, 1].set_title(f'x_hat_{i+1}')

            # finalizar
            fig.suptitle("Señales mezcladas y rescatadas")
            plt.tight_layout()
            plt.show()

        # si es que se tienen las originales
        else:

            # poner x, y, x_hat

            # si no hay tamaño poner
            if figsize is None:
                figsize = (9, self.s*3)
            fig, ax = plt.subplots(self.s, 3, figsize = figsize)

            # ir llenando
            for i in range(self.s):
                # poner la x_i
                ax[i, 0].imshow(señales_x_originales[i], cmap='gray')
                ax[i, 0].axis('off')
                ax[i, 0].set_title(f'x_{i+1}')
                # poner la y_i
                ax[i, 1].imshow(self.señales_y[i], cmap='gray')
                ax[i, 1].axis('off')
                ax[i, 1].set_title(f'y_{i+1}')
                # poner la x_hat_i
                ax[i, 2].imshow(self.x_hat[i], cmap='gray')
                ax[i, 2].axis('off')
                ax[i, 2].set_title(f'x_hat_{i+1}')

            # finalizar
            fig.suptitle("Señales originales, mezcladas y rescatadas")
            plt.tight_layout()
            plt.show()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class FCA_fast(Algoritmo_separacion_imagenes_fast):
    '''
    Separacion de señales con entradas libres.

    Especificaciones:
        Funcional = 1/n Tr
        cumulante_4 = kappa_4 (cumulante libre)
        norma =  valor absoluto
    '''

    # implementar el funcional
    def funcional(self, Z):
        return (1/self.n) * Z.trace()

    # implementar el cumulante
    def cumulante_4(self, Z):
        return self.funcional( np.linalg.matrix_power(Z, 4) ) - 2 * ( self.funcional( np.linalg.matrix_power(Z, 2) ) )**2

    # implemetar la norma
    def norma(self, x):
        return np.abs(x)

# -----------------------------------------------------------------------------------------------------------

class OVFCA_fast(Algoritmo_separacion_imagenes_fast):
    '''
    Separacion de señales con entradas libres con amalgacion sobre B,
    esto con respecto a una esperanza condicional
    F: A -> B
        A: matrices nxn
        B: subconjunto  (subalgebra) de matrices nxn


    Especificaciones:
        Funcional = 1/n Tr
        cumulante_4 = kappa_4^B (cumulante valuado en operadores )
        norma =  norma en B
    '''

    # sobre escribir el cosntructor, para delimitar
    # una esperanza condicional y una norma en su imagen
    def __init__(self, señales_y, esperanza_condicional_F, norma_en_B):
        # llamar al constructor del padre
        super().__init__(señales_y)

        # poner estos como atributos
        self.F = esperanza_condicional_F
        self.norma_en_B = norma_en_B


    # implementar el funcional
    def funcional(self, Z):
        return (1/self.n) * Z.trace()

    # implementar el cumulante
    def cumulante_4(self, Z):
        return self.F( np.linalg.matrix_power(Z, 4) ) - ( self.F(np.linalg.matrix_power(Z, 2)) )**2 - self.F( Z @ self.F(np.linalg.matrix_power(Z, 2)) @ Z)

    # implemetar la norma
    def norma(self, x):
        return self.norma_en_B(x)


# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# ICA
class ICA():
    '''
    Clase de un algoritmo ICA de separacion de señales.
    Basado en la descripcion  en: Free Component Analysis: Theory, Algorithms and Applications por Raj Rao Nadakuditi1, Hao Wu

    Se tienen muestras de s señales y = y1, ..., y_s.
    Las muestras son de tamaño N, entonces las muestras se guardan en una matriz: s x N
    donde cada renglon es la muestra de tamaño N de una señal

    Se supone que y = Ax
    para una matriz desconocida A que es s x s
    y unas señales originales x = x1, ..., x_s

    Se intenta recuperar A y x haciendo ciertas suposiciones
    '''

    # constructor
    def __init__(self, señales_y):

        # poner las señales y como atributo
        self.señales_y = señales_y

        # ver las dimensiones
        self.s = señales_y.shape[0]
        self.N = señales_y.shape[1]

    # ----------------------------------------------------------------------------------------

    # covariancas
    def compute_covarianzas(self):
        '''
        Calcular la matriz sxs de covarianzas
        '''

        # primero centrar todas las y
        self.medias_señales = self.señales_y.mean(axis = 1).reshape(self.s, 1) # column vector
        self.señales_y_centradas = self.señales_y - self.medias_señales @ np.ones(self.N).reshape(1, self.N)

        # calcular covarianzas
        self.C = 1/self.N * self.señales_y_centradas @ self.señales_y_centradas.T

        # ya se tiene C, tambien calcular C^{1/2} y C^{-1/2}
        self.C_power_un_medio = sp.linalg.sqrtm(self.C)
        self.C_power_menos_un_medio = np.linalg.inv(self.C_power_un_medio)

        # devolver C
        return self.C


    def whitening(self):
        """
        Hacer whitening de las señales originales y
        """
        self.señales_y_whitening = self.C_power_menos_un_medio @ self.señales_y_centradas

    # ----------------------------------------------------------------------


    def empirical_kurtosis(self, x):
        """
        Empirical kurtosis of a vector x of N elements
        """
        return 1/self.N * ( np.sum(x**4) ) - 3 * (1/self.N * np.sum(x**2) )**2


    def cantidad_maximizar(self, W):
        '''
        La cantidad que debe de ser maximizada por la matriz W
        evaluada en una matriz especifica
        '''

        # primero calcular el producto de matrices
        producto_matrices = W.T @ self.señales_y_whitening # shape s x N

        # ir calculando cada termino de la suma
        cantidad_final = 0
        for vector in producto_matrices: # iterar en las s señales con whitening
            cantidad_final += np.abs( self.empirical_kurtosis (vector) )

        # devolvere toda la suma
        return cantidad_final


    def vector_to_matrix(self, vector):
        '''
        Pasar de un vector de s^2 elementos
        a una matriz sxs
        '''
        return vector.reshape(self.s, self.s)


    def matrix_to_vector(self, matrix):
        '''
        Pasar de una matriz sxs
        a un vector de s^2 elementos
        '''
        return matrix.flatten()


    def punto_inicial_W(self, iniciar_W_ortogonal):
        '''
        Devuelve un vector con las entradas de W aleatorio
        Usado como valor inicial para la optimizacion

        Se puede forzar a que sea ortogonal,
        usando una descomposicion QR
        '''

        # se quiere ortogoanl
        if iniciar_W_ortogonal:
            # tomar una aleatoria, y descomponer en QR, tomar Q
            Q, _ = np.linalg.qr(np.random.rand(self.s, self.s))
            vector_W_inicial = self.matrix_to_vector(Q)
        # solo aleatoria
        else:
            vector_W_inicial = np.random.rand(self.s**2)

        return vector_W_inicial


    # ----------------------------------------------------------------------

    # Problema de optimizacion
    def resolver_optimizacion(self, repeticiones_optimizacion, iniciar_W_ortogonal):
        '''
        Encuentra la matriz W_hat que maximiza la cantidad deseada
        Resuelve la optimizacion usando Byrd-Omojokun Trust-Region SQP
        '''

        # hacer la funcion a minimizar, toma un vector de s^2 elementos
        def funcion_objetivo_minimizar(vector_W):
            return -1 * self.cantidad_maximizar( self.vector_to_matrix(vector_W) )

        # calcular producto interno para pares de filas de W
        def restriccion_productos_internos(vector_W):
            W = self.vector_to_matrix(vector_W)
            # calcular para cada par de filas (i, j)
            restricciones = []
            for i in range(self.s):
                for j in range(i, self.s):
                    if i == j:
                        # el producto interno debe ser 1
                        restricciones.append(W[i] @ W[i] - 1)
                    else:
                        # el producto interno debe ser 0
                        restricciones.append(W[i] @ W[j])
            return np.array(restricciones)

        # crear la restriccion, esta funcion debe ser 0 en todas sus entradas
        constraint = NonlinearConstraint(restriccion_productos_internos, 0, 0)

        # se va a hacer la optimizacion varias veces, guardar
        self.best_valor_objetivo_opti = float('-inf')  # el mejor, considerando el problema de maximizacion
        self.valores_objetivo_opti = []

        # hacer las iteraciones
        for _ in range(repeticiones_optimizacion):

            # tomar el punto inicial para W
            vector_W_inicial = self.punto_inicial_W(iniciar_W_ortogonal)

            # hacer la minimizacion (mazimizacion)
            results_opti = minimize(funcion_objetivo_minimizar, vector_W_inicial,
                                    constraints= constraint, method = "trust-constr")

            #valor de la función objetivo
            valor_objetivo = -1 * results_opti.fun  # estamos maximizando

            # guardar todos
            self.valores_objetivo_opti.append(valor_objetivo)

            # Guardar si es el mejor resultado hasta ahora
            if valor_objetivo > self.best_valor_objetivo_opti:
                # poner como atributos
                self.best_valor_objetivo_opti = valor_objetivo
                self.best_valores_w_opti = results_opti.x
                self.best_results_opti = results_opti

        # fin de las repeticiones

        # obtener la matriz W_hat
        self.W_hat = self.vector_to_matrix( self.best_valores_w_opti )

        return self.W_hat


    # ----------------------------------------------------------------------
    # METODO PRINCIPAL

    def separar_señaes(self, repeticiones_optimizacion = 3, iniciar_W_ortogonal = True):
        '''
        Metodo principal
        Estima A^{-1} para obtener estimaciones x_hat de x

        Es decir, separa las señales y en x_hat,
        estimando las señales originales desconocidas x

        repeticiones_optimizacion - numero de veces que
        se intenta resolver el problema de optimizacion
        iniciar_W_ortogonal - indicar si el valor inicial para
        la optimizacion es una matriz ortogonal o no
        '''

        # calcular covarianzas y whitening
        self.compute_covarianzas()
        self.whitening()
        # hacer optimizacion para encontar W_hat
        self.resolver_optimizacion(repeticiones_optimizacion, iniciar_W_ortogonal)

        # calcular A_hat
        self.A_hat = self.C_power_un_medio @ self.W_hat
        # invertir
        self.A_hat_inversa = np.linalg.inv(self.A_hat)

        # calular x_hat
        self.x_hat = self.A_hat_inversa @ self.señales_y

        return self.x_hat

# -----------------------------------------------------------------------------------------------------------

# ICA especial para imagenes, con el mismo formato que FCA, BCA

class ICA_imagen():
    '''
    ICA especializado, donde las señales son imagenes

    Internamente se usa in ICA normal, solo es para hacerlo mas facil de usar
    '''

    # constructor
    def __init__(self, señales_y):

        # poner las señales y como atributo
        self.señales_y = señales_y

        # ver cuantas señales son
        self.s = len(señales_y)

        # ver que cada una es n x n
        self.n = señales_y[0].shape[0]
        for y_i in señales_y:
            assert y_i.shape == (self.n, self.n)

        # entonces son vectores de tañano n^2 para el ICA
        self.señales_y_matriz = np.array([señal.flatten() for señal in señales_y])

        # hacer el ICA original, que es el que hace todos los calculos
        self.ICA_original = ICA(self.señales_y_matriz)


    # ----------------------------------------------------------------------

    # funcionaes auxiliares para la optimizacion

    def multuplicar_matrix_con_vector_de_matrices(self, M, z):
        """
        Dado una lista z de s matrices nxn (z pueden ser las señales y)
        y una matriz M que es sxs, calcular la multiplicacion,
        donde se ausme que z es un vector donde sus componentes son matrices.
        Es decir, se hace:

        np.kron(M, np.eye(n)) z

        y se deuevele el resutlado como una lista de matrices (de la forma de z)
        """

        # lista de matrices como resultado
        return [sum(M[i][j] * z[j] for j in range(self.s))
                for i in range(self.s)]

    # ----------------------------------------------------------------------

    # Problema de optimizacion
    def resolver_optimizacion(self, repeticiones_optimizacion, iniciar_W_ortogonal):
        '''
        Encuentra la matriz W_hat que maximiza la cantidad deseada
        Resuelve la optimizacion usando Byrd-Omojokun Trust-Region SQP
        '''

        # hacer la optimizacion con el ICA
        self.ICA_original.resolver_optimizacion(repeticiones_optimizacion, iniciar_W_ortogonal)

        # poner atributos
        self.W_hat = self.ICA_original.W_hat

        return self.W_hat

    # ----------------------------------------------------------------------
    # METODO PRINCIPAL

    def separar_señaes(self, repeticiones_optimizacion = 3, iniciar_W_ortogonal = True):
        '''
        Metodo principal
        Estima A^{-1} para obtener estimaciones x_hat de x

        Es decir, separa las señales y en x_hat,
        estimando las señales originales desconocidas x

        repeticiones_optimizacion - numero de veces que
        se intenta resolver el problema de optimizacion
        iniciar_W_ortogonal - indicar si el valor inicial para
        la optimizacion es una matriz ortogonal o no
        '''

        # calcular covarianzas y whitening
        self.ICA_original.compute_covarianzas()
        self.ICA_original.whitening()
        # hacer optimizacion para encontar W_hat
        self.resolver_optimizacion(repeticiones_optimizacion, iniciar_W_ortogonal)

        # calcular A_hat
        self.A_hat = self.ICA_original.C_power_un_medio @ self.W_hat
        # invertir
        self.A_hat_inversa = np.linalg.inv(self.A_hat)

        # calular x_hat
        self.x_hat = self.multuplicar_matrix_con_vector_de_matrices(self.A_hat_inversa, self.señales_y)

        return self.x_hat

    # ----------------------------------------------------------------------
    # Visualizaciones

    # ver solo x_hat
    def ver_señales_rescatadas(self):
        '''
        Ver las señales x_hat
        '''

        # ver las s señales
        fig, ax = plt.subplots(self.s)

        # por cada i
        for i in range(self.s):
            # dibujar x_hat_i
            ax[i].imshow(self.x_hat[i], cmap='gray')
            ax[i].axis('off')
            ax[i].set_title(f'x_hat_{i+1}')

        # finalizar
        fig.suptitle("Señales rescatadas")
        plt.tight_layout()
        plt.show()

    def ver(self, señales_x_originales = None, figsize = None):
        '''
        Ver señales rescatadas x_hat, señales mezcladas y
        y opcionalmente tambien las señales originales
        '''

        # si no se tienen las originales
        if señales_x_originales is None:

            # solo poner y, x_hat

            # si no hay tamaño poner
            if figsize is None:
                figsize = (6, self.s*3)
            fig, ax = plt.subplots(self.s, 2, figsize = figsize)

            # ir llenando
            for i in range(self.s):
                # poner la y_i
                ax[i, 0].imshow(self.señales_y[i], cmap='gray')
                ax[i, 0].axis('off')
                ax[i, 0].set_title(f'y_{i+1}')
                # poner la x_hat_i
                ax[i, 1].imshow(self.x_hat[i], cmap='gray')
                ax[i, 1].axis('off')
                ax[i, 1].set_title(f'x_hat_{i+1}')

            # finalizar
            fig.suptitle("Señales mezcladas y rescatadas")
            plt.tight_layout()
            plt.show()

        # si es que se tienen las originales
        else:

            # poner x, y, x_hat

            # si no hay tamaño poner
            if figsize is None:
                figsize = (9, self.s*3)
            fig, ax = plt.subplots(self.s, 3, figsize = figsize)

            # ir llenando
            for i in range(self.s):
                # poner la x_i
                ax[i, 0].imshow(señales_x_originales[i], cmap='gray')
                ax[i, 0].axis('off')
                ax[i, 0].set_title(f'x_{i+1}')
                # poner la y_i
                ax[i, 1].imshow(self.señales_y[i], cmap='gray')
                ax[i, 1].axis('off')
                ax[i, 1].set_title(f'y_{i+1}')
                # poner la x_hat_i
                ax[i, 2].imshow(self.x_hat[i], cmap='gray')
                ax[i, 2].axis('off')
                ax[i, 2].set_title(f'x_hat_{i+1}')

            # finalizar
            fig.suptitle("Señales originales, mezcladas y rescatadas")
            plt.tight_layout()
            plt.show()


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Evaluar un algoritmo de separacion de señales

def generate_permutation_matrices(n):
    """Genera todas las matrices de permutación de tamaño n x n."""
    perms = itertools.permutations(range(n))
    P_matrices = []
    for perm in perms:
        P = np.zeros((n, n))
        for i, j in enumerate(perm):
            P[i, j] = 1
        P_matrices.append(P)
    return P_matrices


def optimal_scaling_matrix(A_hat_inv, A, P):
    """Calcula la matriz diagonal óptima D minimizando || P D A_hat_inv A - I ||_F."""
    n = A.shape[0]

    def loss(D_vec):
        D = np.diag(D_vec)
        return scipy.linalg.norm(P @ D @ A_hat_inv @ A - np.eye(n), 'fro')

    D_init = np.ones(n)  # Inicializar D como identidad
    result = minimize(loss, D_init, method='Powell')  # Optimización
    return np.diag(result.x)  # Devolver matriz diagonal óptima


def unmixing_error(A, A_hat):
    """
    Calcula el error de separación
    A es la matriz de mexcla original
    A_hat es la estimacion por el metodo de separacion ciega de señales
    """
    A_hat_inv = scipy.linalg.inv(A_hat)
    P_matrices = generate_permutation_matrices(A.shape[0])

    min_error = float('inf')
    for P in P_matrices:
        D_opt = optimal_scaling_matrix(A_hat_inv, A, P)
        error = scipy.linalg.norm(P @ D_opt @ A_hat_inv @ A - np.eye(A.shape[0]), 'fro')
        min_error = min(min_error, error)

    return min_error

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

