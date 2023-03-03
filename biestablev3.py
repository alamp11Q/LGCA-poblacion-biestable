# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:25:57 2022

@author: alamp
"""

import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import statistics

random.seed(18)


# %% -----------------------FUNCIONES---------------------
#--------------------Modelo LGCA--------------------
def propagacion(nodos):

        nuevosnodos = np.zeros(nodos.shape, dtype=nodos.dtype)
        nuevosnodos[..., 2:] = nodos[..., 2:]
        nuevosnodos[...,:-1, 0] = nodos[...,1:, 0]
        nuevosnodos[...,1:, 1] = nodos[...,:-1, 1]

        return nuevosnodos
  

def desordenar(a, axis=-1):
    b = a.swapaxes(axis, -1)
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return

def interacciones(nodos,k,n,prob=1):
    n_x = np.array([nodos[...,].sum(-1)])
    rho_x = n_x/k
    p_nacimiento=0
    p_muerte=0
    
    
    if prob == 1:
        p_muerte = 10*rho_x*(0.5 - 0.8*rho_x)**2
        p_nacimiento = rho_x*(1 - rho_x)
    
    elif prob == 2:
        R=7
        a=0.4
           
        p_nacimiento=rho_x*(1-rho_x)
        p_muerte=R*rho_x**3-(3*a*R+1)*rho_x**2+(2*R*a**2+1)*rho_x
    
    var_al = npr.random(n_x.shape) 
    
    #f = p_n - p_m
    dn_x = (var_al < p_nacimiento).astype(np.int8)
    dn_x -= np.logical_and(p_nacimiento < var_al, var_al < (p_muerte + p_nacimiento))

    n_x += dn_x
    nuevosnodos = np.zeros(nodos.shape, dtype=nodos.dtype)

    comb = [list(range(0,1)),list(range(0,n))]
    coord_pairs = list(itertools.product(*comb))
    

    for coord in coord_pairs:
        nuevosnodos[coord[1],:n_x[coord]] = 1

    newv = nuevosnodos[...,]
    desordenar(newv, axis=-1)
    nuevosnodos[...,] = newv


    return nuevosnodos


def quimografia(matriz,iteraciones,n,k,tao=1,e=1):
    size=14
    mat_graf = matriz[...,:].sum(-1)/k
    
    fig, ax = plt.subplots()
    
    ax.set_xlabel('x',fontsize=size)
    ax.set_ylabel('t',fontsize=size)
    ax.set_title('Dinámica de una población.',fontsize = size)
    
    ax.set_xticks(list(range(0, n+1,10)))
    ax.set_xticklabels(np.array(list(range(0, n+1,10)))*e)
    
    ax.set_yticks(list(range(0, iteraciones+1,200)))
    ax.set_yticklabels(np.array([round(i*tao,2) for i in list(range(0, iteraciones+1,200))]))
    
    c = ax.pcolormesh(mat_graf, cmap='hot', vmin=0, vmax=1)
    cb = fig.colorbar(c)

    cb.ax.tick_params(axis='y', which='major', labelsize=size)
    cb.set_label('Densidad de la población', fontsize=size)
    #fig.colorbar(c, ax=ax)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.show()
    
def grafica_biestable(matriz,k,n,iteracion,e):
    size = 18
    mat_graf = matriz[iteracion].sum(-1)/k 
    x = np.array([i for i in range(1,n+1)])*e
    fig, ax1 = plt.subplots()
    

    ax1.plot(x,mat_graf)

    ax1.set_xlabel('x',fontsize=size)
    ax1.set_ylabel(r'$\rho$(x,t)',fontsize=size,rotation =1)
    ax1.set_ylim(0,1)

    ax1.xaxis.set_label_coords(1.025, 0.020)
    ax1.yaxis.set_label_coords(0.001, 1.025)
    
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.show() 

def biestable(nodos,k,iteraciones,n,espacio=True,prob=1):
    matriz = np.zeros((iteraciones+1,n,k), dtype=bool)
    matriz[0] = nodos
    if espacio == True:
        for i in range(1,iteraciones+1):
            anterior = matriz[i-1]
            nodos1 = interacciones(anterior,k,n,prob)
            nodos2 = propagacion(nodos1)
            matriz[i] = np.array([nodos2])
    else:
          for i in range(1,iteraciones+1):
              anterior = matriz[i-1]
              nodes1 = interacciones(anterior,k,n,prob)
              matriz[i] = np.array([nodes1])
    return matriz
    
def vel_prop(k,D,u1,u2,u3):
    return np.sqrt(k*D/2)*(u1 + u3 - 2*u2)

#-----------------Cadena de Markov------------------
def MatrizTransicion(k):
    T = np.zeros([k+1,k+1])
    for i in range(k+1):
        p_muerte = 10*(i/k)*(0.5 - 0.8*(i/k))**2
        p_nacimiento = (i/k)*(1 - (i/k))
        if i == 0:
            T[i,i] = 1
        elif i == k:
            T[i,i] = 1 - p_muerte
            T[i,i-1] = p_muerte
        else:
            T[i,i-1] = p_muerte
            T[i,i] = 1 - p_nacimiento - p_muerte
            T[i,i+1] = p_nacimiento
    return T

def MatrizTransicion2(k):
    T = np.zeros([k+1,k+1])
    for i in range(k+1):
        p_muerte = 10*(i/k)*(0.5 - 0.8*(i/k))**2
        p_nacimiento = (i/k)*(1 - (i/k))
        if i == 0:
            T[i,i] = 1
        elif i == k:
            T[i,i] = 1 - p_muerte
            T[i-1,i] = p_muerte
        else:
            T[i-1,i] = p_muerte
            T[i,i] = 1 - p_nacimiento - p_muerte
            T[i+1,i] = p_nacimiento
    return T

#Calculamos la matriz de transicion multiplicada n veces (T^n)
def Tn(T,n):
    Tn = T.copy()
    for i in range(1,n):
        Tn = np.dot(Tn,T)
    return Tn

#Condicion inicial P0
def P0(k,m):
    T0 = np.zeros(k+1)
    T0[m] = 1
    return T0
    
#Calcular la esperanza del proceso
def EXt(P0,T,t):
    EXt = 0
    if t == 0:
        for i in range(len(P0)):
            EXt += i*P0[i]
    elif t == 1:
        P1 = np.dot(P0,T)
        for i in range(len(P0)):
            EXt += i*P1[i]
    else:
        Tt = Tn(T,t)
        Pt = np.dot(P0,Tt)
        for i in range(len(P0)):
            EXt += i*Pt[i]
    return EXt
    

def grafica_esperanza(P0,T,k,t,tao=1):
    E = np.array([EXt(P0,T,i) for i in range(t+1)])
    t = np.array([i*tao for i in range(t+1)])
    plt.plot(t,E)
    plt.ylabel(r'$\rho$(t)')
    plt.xlabel('t')
    plt.ylim(0,k)
    plt.show()

def grafica_esperanza_normalizada2(P0,T,k,t,tao=1):
    E = np.array([EXt(P0,T,i)/k for i in range(t+1)])
    t = np.array([i*tao for i in range(t+1)])
    plt.plot(t,E)
    plt.ylabel(r'$\rho$(t)',fontsize=16)
    plt.xlabel('t',fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    
    plt.ylim(0,1)
    plt.show()
    
def grafica_esperanza_normalizada(P0,T,k,t,tao=1):
    E = np.array([EXt(P0,T,i)/k for i in range(t+1)])
    t = np.array([i*tao for i in range(t+1)])
    fig, ax1 = plt.subplots()
    ax1.plot(t,E)
    
    # common x axis
    ax1.set_xlabel('t',fontsize=16)
    # First y axis label
    ax1.set_ylabel(r'$\rho$(t)',fontsize=16,rotation =1)
    ax1.set_ylim(0,1)
    
    # Adjust the label location
    ax1.xaxis.set_label_coords(1.025, 0.020)
    ax1.yaxis.set_label_coords(0.001, 1.025)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.show()
    
    
    

def tiempo_promedio_absorcion(T):
    B = T[1:,1:]
    identidad = np.identity(B.shape[0])
    F = np.linalg.inv(identidad - B)
    unos = np.ones(B.shape[0])
    return np.dot(F,unos)
    
def prob_absorcion(X0,k,t,n):
    cont = 0
    for i in range(n):
        Xt = X0
        
        for i in range(t):
            rho_x = Xt/k
            p_muerte = 10*rho_x*(0.5 - 0.8*rho_x)**2
            p_nacimiento = rho_x*(1 - rho_x)
            Xt += (npr.random(1)[0] < p_nacimiento).astype(np.int8)
            Xt -= npr.random(1)[0] < p_muerte
            if Xt == 0:
                cont += 1
                break

    return cont/n

def comparacion_esperanza(P0,T,t,k,nodes,n):
    esperanza = EXt(P0,T,t)
    esperanza_lgca = 0
    for i in range(100):
        LGCA = biestable(nodes,k,t,n,espacio=False)
        esperanza_lgca += LGCA[t].sum(-1)[int(n/2)]/k
    
    print("---------------")
    print("Esperanza(Cantidad de particulas): " + str(esperanza))
    print("Esperanza(Normalizada): " + str(esperanza/k))
    print("Estimacion LGCA: "+str(esperanza_lgca/100))
    

def tiempo_promedio_absorcion_simulacion(X0,k,t,n):
    sim =[]
    while len(sim) < n:
        Xt = X0
        for j in range(1,t+1):
            rho_x = Xt/k
            p_muerte = 10*rho_x*(0.5 - 0.8*rho_x)**2
            p_nacimiento = rho_x*(1 - rho_x)
            var_al = npr.random(1)[0] 
            Xt += (var_al< p_nacimiento).astype(np.int8)
            Xt -= p_nacimiento < var_al < p_muerte + p_nacimiento
            if Xt == 0:
                sim.append(j)
                break
                
    return statistics.mean(sim), statistics.stdev(sim)/np.sqrt(len(sim)),sim


