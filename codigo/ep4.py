"""
=====================================
Trabalho final - EP4 
MAP2210 Aplicações de Álgebra Linear
Author: Pedro Blöss Braga

1o semestre de 2020
=====================================
"""


import time 
from multiprocessing import Manager, Pool 
import multiprocessing as mp
import statistics as stat

import numpy as np
import pandas as pd

import os, sys

#from IPython.display import display
import matplotlib.pyplot as plt
plt.style.use('seaborn')




##############################

import numpy as np
from numpy.linalg import norm

from random import normalvariate
from math import sqrt


def gera_matriz_Hilbert(n):
  """ 
  Gerador de Matriz de Hilbert := H_ij = 1/((i+1)+(j+1)-1) 
  params: 
    n : Tamanho da matriz quadrada
  """
  H=[]
  for j in range(n):
    H.append([])
    for i in range(n):
      H[j].append(1/((i+1)+(j+1)-1))
  return(np.array(H))



def gera_vetor_unitario(n):
  u = [normalvariate(0,1) for _ in range(n)]
  norma = sqrt(sum(x * x for x in u))
  return [x / norma for x in u]



def svd_1D(A, eps=1e-10, verbose=False):

  n,m = A.shape

  x = gera_vetor_unitario(n)
  
  V_ = None
  V = x

  if n>m:
    B = np.dot(A.T, A)
  else:
    B = np.dot(A, A.T)

  it =0

  while True:
    it+=1
    V_ = V

    V = np.dot(B, V_)
    V = V / norm(V) # unitario


    TOL = 1 - eps
    # ||<v_{k}, v_{k-1}>||
    criterio = abs(np.dot(V, V_))
    if criterio > TOL:
      if verbose == True:
        print(f" Método convergiu em {it} iterações. \n ")
      return V

def svd(A, k=None, eps=1e-10, verbose=True):
  
  A = np.array(A, dtype=float)

  n, m = A.shape
  svd_ = []

  if k is None:
    k = min(n, m)
  if verbose == True:
    print(f" k : {k} \n ")

  for i in range(k):
    mat = A.copy()

    for valor_singular, u, v in svd_[:i]:
      mat -= valor_singular * np.outer(u, v)

    if n >m:
      v = svd_1D(mat, eps=eps) # proximo vetor singular
      if verbose == True:
        print(f" prox vetor singular: {v} \n ")
      u_normalizado = np.dot(A, v)
      sigma = norm(u_normalizado) # proximo valor singular
      if verbose == True:
        print(f" prox valor singular: {sigma} \n ")
      u = u_normalizado / sigma

    else:
      u = svd_1D(mat, eps=eps) # proximo vetor singular
      if verbose == True:
        print(f" prox vetor singular: {u} \n ")
      v_normalizado = np.dot(A.T, u)
      sigma = norm(v_normalizado) # próximo valor singular
      if verbose == True:
        print(f" prox valor singular: {sigma} \n ")
      v = v_normalizado / sigma

    svd_.append((sigma, u, v))

  valores_singulares, us, vs = [np.array(x) for x in zip(*svd_)]
  grade = 60*'='
  if verbose == True:
    print(
        f" {grade} \n valores singulares: {valores_singulares} \n ",
        f" U: {us} \n ",
        f" V: {vs} \n {grade}"
          )
  return valores_singulares, us.T, vs



def compara_determinantes(A, U, S, V, verbose=False):
  try:
    m = len(A)
    Z = np.array([np.zeros(m)]*m)

    for k in range(m):
      Z[k][k] = S[k]

    A_ = np.array(np.matmul(U, np.inner(Z, V.T)))
    if verbose == True:
      print(f" U Z V^T: {A_} \n ")
    
    det_A_ = np.linalg.det(A_)
    det_A = np.linalg.det(A)
    return abs(det_A_ - det_A)
  except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(f"Z: {Z} \n S[k]: {S[k]} \n ")
      print(f" Erro no (compara_determinantes) : \n {e} \n exc_type: {exc_type} , fname: {fname} , exc_tb.tb_lineno: {exc_tb.tb_lineno} \n ")    


def aplica_teste(
    A, 
    shared_list_linalg_tempo,
  shared_list_svd_tempo,
  shared_list_linalg_dif,
  shared_list_svd_dif
    ):
  assert A.shape[0] == A.shape[1], " Matriz deve ser quadrada para performar teste, pois calcularei determinante."
  try:
    
    ###############################################
    # teste com linalg
    t0 = time.time()
    U0, S0, V0 = np.linalg.svd(A)
    dt0 = time.time() - t0


    dif0 = compara_determinantes(A, U0, S0, V0)
    #dif0= 0
    ###############################################
    
    ###############################################
    # teste com implementação 
    print("Aplicando teste... ")
    t1 = time.time()
    S1, U1, V1 = svd(A, verbose=False)
    dt1 = time.time() - t1
    dif1=0
    print("Aplicando teste... ")
    
    '''
    try:
      dif1 = compara_determinantes(A, U1, S1, V1)
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
      print(f" Erro no (compara_Dets 2) : \n {e} \n exc_type: {exc_type} , fname: {fname} , exc_tb.tb_lineno: {exc_tb.tb_lineno} \n ")    
      dif1 = 0
    '''
    dif1 = compara_determinantes(A, U1, S1, V1)
    ###############################################
    

    print(f" dt0: {dt0} dt1: {dt1} \n dif0: {dif0} dif1: {dif1} \n ")
    shared_list_linalg_tempo.append(dt0)
    shared_list_svd_tempo.append(dt1)

    shared_list_linalg_dif.append(dif0)
    shared_list_svd_dif.append(dif1)
  except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f" Erro no (aplica_teste) : \n {e} \n exc_type: {exc_type} , fname: {fname} , exc_tb.tb_lineno: {exc_tb.tb_lineno} \n ")    



def testes_matrizes_quadradas_svd(K=20, tipo='hilbert'):

  # iniciadores do multiprocessamento
  manager = Manager()
  print(f"CPUs: {mp.cpu_count()} \n ")
  pool = Pool(processes=mp.cpu_count())

  shared_list_linalg_tempo = manager.list()
  shared_list_svd_tempo = manager.list()
  shared_list_linalg_dif = manager.list()
  shared_list_svd_dif = manager.list()


  # números dos tamanhos das matrizes quadradas que irei testar
  r = list(range(2, K, 2))
  c=0
  for k in r: 
    # gera matriz pseudo-aleatória ou de hilbert
    if tipo == 'hilbert':
      A = gera_matriz_Hilbert(k)
    else:
      A = np.random.rand(k, k)

    # adiciona teste ao pool
    pool.apply_async(
        aplica_teste,
        [
         A,
         shared_list_linalg_tempo,
        shared_list_svd_tempo,
        shared_list_linalg_dif,
        shared_list_svd_dif
        ]
    )
    c+=1
    print("({}%) ({} de {}) (pool {})".format(
        c*100/(K//2), c, K//2, pool
    ))

  pool.close()
  pool.join()


  tempos_linalg = list(shared_list_linalg_tempo)
  tempos_svd = list(shared_list_svd_tempo)
  difs_linalg = list(shared_list_linalg_dif)
  difs_svd = list(shared_list_svd_dif)

  print(
      f"  tempos_linalg: {tempos_linalg} \n ",
      f" tempos_svd: {tempos_svd} \n ",
      f" difs_linalg: {difs_linalg} \n ",
      f" difs_svd: {difs_svd} \n ",
  )
  d = {
      'Média tempos': [stat.mean(tempos_linalg), stat.mean(tempos_svd)],
       'Média ($||det(A) - det(A*)||_2$)': [stat.mean(difs_linalg), stat.mean(difs_svd)]
  }   
  df = pd.DataFrame(
      data = d, 
      columns = list(d.keys()),
      index = ['linalg', 'próprio']
  )

  #display(df)
  print(df)


  plt.figure(figsize=(15,8))

  plt.rcParams['font.size'] = 25
  plt.subplot(2,1,1)
  plt.title('Tempos', fontsize=25)
  plt.scatter(r, tempos_linalg, color='red', s=35)
  plt.plot(r, tempos_linalg, label='linalg')
  plt.scatter(r, tempos_svd, color='green', s=35)
  plt.plot(r, tempos_svd, label='proprio')
  plt.xlabel('nº de linhas')
  plt.legend(loc='best', prop={'size': 25})

  plt.subplot(2,1,2)
  plt.title('||det(A)-det(A^*)||_2', fontsize=25)
  plt.scatter(r, difs_linalg, color='red', s=35)
  plt.plot(r, difs_linalg, label='linalg')
  plt.scatter(r, difs_svd, color='green', s=35)
  plt.plot(r, difs_svd, label='proprio')
  plt.xlabel('nº de linhas')
  plt.legend(loc='best', prop={'size': 25})

  plt.tight_layout()
  plt.show()


  return df

def main():
    print('Iniciando...')
    df = testes_matrizes_quadradas_svd()

if __name__ == "__main__":
    print(__doc__)
    main()