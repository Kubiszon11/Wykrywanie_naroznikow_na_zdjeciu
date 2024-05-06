import cv2
import numpy as np
import matplotlib.pyplot as plt


obraz = cv2.imread('sto.jpg')

szerokosc = obraz.shape[0]
wysokosc = obraz.shape[1]

def podzial_kolorow():
    kolor_r = np.zeros((szerokosc, wysokosc))
    kolor_g = np.zeros((szerokosc, wysokosc))
    kolor_b = np.zeros((szerokosc, wysokosc))

    for i in range(szerokosc):
        for j in range(wysokosc):
            kolor_r[i, j] = obraz[i, j, 0]
            kolor_g[i, j] = obraz[i, j, 1]
            kolor_b[i, j] = obraz[i, j, 2]
    return kolor_r,kolor_g,kolor_b

def pochodna_x(kolor_r, kolor_g, kolor_b):
    kolor_r2 = np.zeros((szerokosc - 1, wysokosc))
    kolor_g2 = np.zeros((szerokosc - 1, wysokosc))
    kolor_b2 = np.zeros((szerokosc - 1, wysokosc))
    for i in range(1,szerokosc - 1):
        for j in range(wysokosc):
            temp1 = -kolor_r[i+1, j] + kolor_r[i - 1, j]
            kolor_r2[i, j] = temp1

            temp1 = -kolor_g[i + 1, j] + kolor_g[i - 1, j]
            kolor_g2[i, j] = temp1

            temp1 = -kolor_b[i+1, j] + kolor_b[i - 1, j]
            kolor_b2[i, j] = temp1
    return kolor_r2, kolor_g2, kolor_b2

def pochodna_y(kolor_r, kolor_g, kolor_b):
    kolor_r2 = np.zeros((szerokosc, wysokosc - 1))
    kolor_g2 = np.zeros((szerokosc, wysokosc - 1))
    kolor_b2 = np.zeros((szerokosc, wysokosc - 1))
    for i in range(szerokosc):
        for j in range(1,wysokosc-1):
            temp1 = -kolor_r[i, j+1] + kolor_r[i, j-1]
            kolor_r2[i, j] = temp1

            temp1 = -kolor_g[i, j + 1] + kolor_g[i, j - 1]
            kolor_g2[i, j] = temp1

            temp1 = -kolor_b[i, j+1] + kolor_b[i, j - 1]
            kolor_b2[i, j] = temp1
    return kolor_r2, kolor_g2, kolor_b2


def narozniki(kolor_r, kolor_g, kolor_b):
    kolor_r2 = np.zeros((szerokosc, wysokosc))
    kolor_g2 = np.zeros((szerokosc, wysokosc))
    kolor_b2 = np.zeros((szerokosc, wysokosc))
    kolor_r3 = np.zeros((szerokosc-2, wysokosc-2))
    kolor_g3 = np.zeros((szerokosc-2, wysokosc-2))
    kolor_b3 = np.zeros((szerokosc-2, wysokosc-2))
    pochX_kolor_r = np.zeros((szerokosc, wysokosc))
    pochY_kolor_r = np.zeros((szerokosc, wysokosc))
    pochX2_kolor_r = np.zeros((szerokosc, wysokosc))
    pochY2_kolor_r = np.zeros((szerokosc, wysokosc))
    pochXY_kolor_r = np.zeros((szerokosc, wysokosc))

    zmiana = np.zeros((szerokosc-2, wysokosc-2))

    for i in range(szerokosc):
        for j in range(wysokosc):
            kolor_r2[i, j] = kolor_r[i, j] * 0.299 + kolor_g[i, j] * 0.587 + kolor_b[i, j] * 0.114
            kolor_g2[i, j] = kolor_r[i, j] * 0.299 + kolor_g[i, j] * 0.587 + kolor_b[i, j] * 0.114
            kolor_b2[i, j] = kolor_r[i, j] * 0.299 + kolor_g[i, j] * 0.587 + kolor_b[i, j] * 0.114
    for i in range(szerokosc-1):
        for j in range(wysokosc-1):
            temp1 = kolor_r[i,j]-kolor_r[i+1,j]
            pochX_kolor_r[i,j]=temp1
            temp1 = kolor_r[i, j] - kolor_r[i, j + 1]
            pochY_kolor_r[i, j] = temp1
    T = -1055000
    k = 0.05
    for i in range(szerokosc-2):
        for j in range(wysokosc-2):
            temp1 = pochX_kolor_r[i,j]-pochX_kolor_r[i+1,j]
            pochX2_kolor_r[i,j]= temp1
            temp1 = pochY_kolor_r[i, j] - pochY_kolor_r[i, j + 1]
            pochY2_kolor_r[i, j] = temp1
            temp1 = pochX_kolor_r[i, j] - pochX_kolor_r[i, j + 1]
            pochXY_kolor_r[i, j] = temp1
            det = pochX2_kolor_r[i, j] * pochY2_kolor_r[i, j] - pochXY_kolor_r[i, j] ** 2
            tr = (pochX2_kolor_r[i, j] * pochY2_kolor_r[i, j]) ** 2
            zmiana[i,j] = det - k * tr
            if zmiana[i,j]<T:
                kolor_r3[i,j]=0
                kolor_g3[i, j] = 0
                kolor_b3[i, j] = 255
                zmiana[i,j]=1
            elif i!=(szerokosc-3) and zmiana[i+1,j]<T:
                kolor_r3[i,j]=0
                kolor_g3[i, j] = 0
                kolor_b3[i, j] = 255
                zmiana[i, j] = 1
            elif i!=0 and zmiana[i - 1, j] < T:
                kolor_r3[i, j] = 0
                kolor_g3[i, j] = 0
                kolor_b3[i, j] = 255
                zmiana[i, j] = 1
            elif j!=(wysokosc-3) and zmiana[i,j+1]<T:
                kolor_r3[i,j]=0
                kolor_g3[i, j] = 0
                kolor_b3[i, j] = 255
                zmiana[i, j] = 1
            elif j!=0 and zmiana[i,j-1]<T:
                kolor_r3[i,j]=0
                kolor_g3[i, j] = 0
                kolor_b3[i, j] = 255
                zmiana[i, j] = 1
            else:
                kolor_r3[i,j]=kolor_r[i,j]
                kolor_g3[i, j] = kolor_g[i,j]
                kolor_b3[i, j] = kolor_b[i,j]





    return kolor_r3, kolor_g3, kolor_b3
def wyswietl_wynik(kolor_r2,kolor_g2,kolor_b2):
    plt.subplot(121), plt.imshow(cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)), plt.title('Oryginalny Obraz')
    plt.subplot(122), plt.imshow(cv2.merge([kolor_r2.astype(np.uint8), kolor_g2.astype(np.uint8), kolor_b2.astype(np.uint8)])), plt.title('Zmieniony Obraz')
    plt.show()

kolor_r,kolor_g,kolor_b = podzial_kolorow()

kolor_r2,kolor_g2,kolor_b2 = narozniki(kolor_r,kolor_g,kolor_b)
wyswietl_wynik(kolor_b2,kolor_g2,kolor_r2)


