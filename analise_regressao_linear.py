import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# leitura do arquivo de entrada
file = sys.argv[1]
dados = pd.read_csv(file)

# criacao dos vetores correspondentes as
# variaveis aleatorias X e Y
x = np.array(dados.x)
y = np.array(dados.y)

# calculo da regressao linear
res = stats.linregress(x, y)

# criacao do grafico de saida
fig,ax = plt.subplots(1)

# diagrama de dispersao
ax.tick_params(direction='in')
font1 = {'family':'serif','size':12}
font2 = {'family':'serif','size':10}
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel("x", fontdict = font2)
plt.ylabel("y", fontdict = font2)
plt.title(file)
plt.plot(x, y, 'o', label='dados originais')
plt.savefig("dispersao.png")

# diagrama de dispersao com a curva de regressao
plt.clf()
ax.tick_params(direction='in')
font1 = {'family':'serif','size':12}
font2 = {'family':'serif','size':10}
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel("x", fontdict = font2)
plt.ylabel("y", fontdict = font2)
plt.title(file+f"\nR\u00b2={res.rvalue**2:.2f}     \
          \u03B2\u2080={res.intercept:.2f}\u00B1{res.intercept_stderr:.2f}     \
          \u03B2\u2081={res.slope:.2f}\u00B1{res.stderr:.2f}", fontdict = font1)
plt.plot(x, y, 'o', label='dados originais')
plt.plot(x, res.slope * x + res.intercept, 'r', label='linha ajustada')
plt.savefig("regressao.png")

# grafico dos residuos em funcao de x
plt.clf()
ax.tick_params(direction='in')
font1 = {'family':'serif','size':12}
font2 = {'family':'serif','size':10}
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel("x", fontdict = font2)
plt.ylabel("y", fontdict = font2)
plt.title(file+f"\nR\u00b2={res.rvalue**2:.2f}     \
          \u03B2\u2080={res.intercept:.2f}\u00B1{res.intercept_stderr:.2f}     \
          \u03B2\u2081={res.slope:.2f}\u00B1{res.stderr:.2f}", fontdict = font1)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlabel("x", fontdict = font2)
plt.ylabel("$e_i$", fontdict = font2)
plt.plot(x, y - (res.slope * x + res.intercept), 'o', label='resíduos')
plt.axhline(y=0.0, color='r', linestyle='-')
plt.savefig("residuos.png")