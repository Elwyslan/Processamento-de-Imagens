"""
ALUNO: Jose Elwyslan Mauricio de Oliveira

MATRICULA: 201110005474

DATA: 28/08/2014

PROFESSOR: Daniel de Oliveira Dantas

DESCRICAO:

Implementacao da segmentacao dos nucleos das celulas brancas do sangue (WBC - white blood cells) proposto por H. T. Madhloom em :

"An Automated White Blood Cell Nucleus Localization and Segmentation using Image Arithmetic and Automatic Threshold,"
Journal of Applied Sciences, vol. 10, no. 11, pp. 959-966, 2010.

Obs: O algoritmo teve alguns de seus passo modificados.

Foi passado um dataset com 367 imagens de celulas sanguineas que pode ser baixado e acessado atraves do link:
http://www.mathworks.com/matlabcentral/fileexchange/36634-an-efficient-technique-for-white-blood-cells-nuclei-automatic-segmentation
"""

"""
A biblioteca OpenCV para Windows pode se baixada atraves do link:
	http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

A biblioteca Numpy para Windows pode se baixada atraves do link:
	http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy

A biblioteca Scipy para Windows pode se baixada atraves do link:
	http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy

Para o Ubuntu os itens necessarios sao:

[Compilador] sudo apt-get install build-essential

[Requerido]  sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

[Opcional]   sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev


Obs: Este codigo foi testado somente no Windows 8

"""
import cv2

import numpy as np

import os

from scipy.ndimage.filters import minimum_filter as minfilter

from scipy.ndimage.measurements import label

""" Funcoes Auxiliares """
#Retorna as dimensoes da imagem
def size(image):
	return np.array([image.shape[1], image.shape[0]], long)

#Exibe uma imagem utilizando o openCV
def cv2imshow(image, windowName='image'):
	cv2.imshow(windowName,image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Mascara utilizada na abertura morfologica do passo 10
mascara= np.array([[0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
				   [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
				   [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
				   [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
				   [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
				   [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
				   [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
				   [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0]], np.uint8)

""" Inicio das funcoes auxiliares do algoritimo """

#1) Convert the input image, A, to a gray scale image B.
def imreadgray(filename):
	return cv2.imread(filename,0)


#2)  Adjust the gray scale image, B, intensity values with a linear contrast stretching to get image L. 
def linearContrastStretching(image,new_limitInf = 0.0, new_limitSup = 255.0):
	image = image.astype(np.float32)#Muda o tipo de dados da imagem de entrada pra float

	#Pega a maior e a menor intensidade presentes na imagem
	imgMaxValue = np.amax(image)
	imgMinValue = np.amin(image)

	#Pega as dimensoes da imagem
	imgWidth, imgHeight = size(image)

	#Gera a imagem de saida
	outputImage = np.zeros((imgHeight,imgWidth), np.uint8)

	#Calculo da constante 'k' para fazer o ajuste linear do contraste que obdece a equacao [y = kx + c]
	aux = (new_limitSup - new_limitInf)/(imgMaxValue - imgMinValue)

	#Faz o ajuste. Os pixels da imagem de saida estaram agora entre 'new_limitInf' e 'new_limitSup'
	for i in range(0, imgHeight):
		for j in range(0, imgWidth):
			#Mapeia o valor do pixel de entrada para o novo valor do pixel de saida
			newValue = round(aux * (image.item(i,j) - imgMinValue) + new_limitInf)
			
			#Trata a saturacao
			if newValue>new_limitSup:
				newValue = new_limitSup
			if newValue<new_limitInf:
				newValue = new_limitInf

			#Atribui o valor do pixel a imagem de saida
			outputImage.itemset((i,j), int(newValue))

	#retorna o tipo de dados da imagem de entrada para unsigned int de 8 bits
	image = image.astype(np.uint8)

	#retorna a imagem de saida
	return outputImage

#3) Enhance the contrast of the gray scale image, B, using histogram equalization to get image H. 
def histeq(image):
	return cv2.equalizeHist(image)
	
#4) Obtain the image R1=L+H.
def imadd(image1, image2):

	image1 = image1.astype(np.int32)
	image2 = image2.astype(np.int32)

	outputImage = image1.__add__(image2)

	imgWidth, imgHeight = size(outputImage)

	for i in range(0, imgHeight):
		for j in range(0, imgWidth):
			if outputImage.item(i,j)>255:
				outputImage.itemset((i,j), 255)

	outputImage = outputImage.astype(np.uint8)

	image1 = image1.astype(np.uint8)
	image2 = image2.astype(np.uint8)

	return outputImage

#5) Obtain the image R2=L-H.
def imsubtract(image1, image2):
	image1 = image1.astype(np.int32)
	image2 = image2.astype(np.int32)

	outputImage = image1.__sub__(image2)

	imgWidth, imgHeight = size(outputImage)

	for i in range(0, imgHeight):
		for j in range(0, imgWidth):
			if outputImage.item(i,j)<0:
				outputImage.itemset((i,j), 0)

	outputImage = outputImage.astype(np.uint8)
	
	image1 = image1.astype(np.uint8)
	image2 = image2.astype(np.uint8)

	return outputImage

#6)  Obtain the image R3=R1+R2
#Ja foi feita uma funcao para isso no passo 4

#7)  Implement, three times, 3-by-3 minimums filter on the image R3.
def minimumfilter(image, mask = np.ones((3,3))):
	return minfilter(image, footprint=mask)

#8)  Calculate a global threshold value using Otsu's method.
#9)  Convert R3 to binary image using the threshold from step 8.
def otsuglobalthreshold(image):
	ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	
	return th2

#9.1)Inverter a imagem binaria
def invertImage(image):
	return image.__invert__()

#10) Use morphological opening to remove small pixel groups. Use a disk structuring element with a radius of 9 pixels.
def morphologicalopening(image, mask):
	return cv2.morphologyEx(image, cv2.MORPH_OPEN, mask)

#11) Connect the neighboring pixels to form objects.
def connectneighboringpixels(binaryImage):
	#Pega as dimensoees da imagem
	width, height = size(binaryImage)

	#Indentifica todos os componentes conexos da imagem
	labeled, n = label(binaryImage, np.ones((3,3)))
	#Modifica o tipo de dados para unsigned int de 8 bits
	labeled = labeled.astype(np.uint8)

	width, height = size(labeled)

	for i in range(0, height-1):
		for j in range(0,width-1):
			if labeled.item(i,j) > 0:
				labeled.itemset((i,j), 255)
	return invertImage(labeled)

#11) Connect the neighboring pixels to form objects.
def removeIrrelevantObjects(binaryImage):
	#Pega as dimensoees da imagem
	width, height = size(binaryImage)

	#Indentifica todos os componentes conexos da imagem
	labeled, n = label(binaryImage, np.ones((3,3)))

	#Modifica o tipo de dados para unsigned int de 8 bits
	labeled = labeled.astype(np.uint8)

	l = list()
	for areas in range(1, n+1):
		area = (labeled == areas)#Indentifica os pixel de uma determinada area
		pixels = np.extract(area, labeled)#Pega os pixel de uma determinada area
		l.append(pixels.size)#Armazena a quantidade de pixels de cada area na lista

	#(1 pixel) == (1 unidade de area)
	#Elimina objetos com area menor do que 5000 (5000 pixels)
	width, height = size(labeled)
	for i in range(0, len(l)):
		if l[i] < 5000:
			areaToEliminate = i+1
			for j in range(0, height):
				for k in range(0,width):
					if labeled.item(j,k) == areaToEliminate:
						labeled.itemset((j,k), 0)

	for i in range(0, height):
		for j in range(0,width):
			if labeled.item(i,j) > 0:
				labeled.itemset((i,j), 255)
	return invertImage(labeled)

def hsv_Processing(image):
	#Converte a imagem para o sistema de cores HSV
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	#Range da cor Azul no HSV
	lower_blue = np.array([110,50,50], dtype=np.uint8)
	upper_blue = np.array([130,255,255], dtype=np.uint8)

	#Cria um mascara do tamanho da imagem somente com os pixels da cor azul
	binary_bluePixels = cv2.inRange(hsv, lower_blue, upper_blue)

	#Cria a imagem de saida em escala de cinza
	outputImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	width, height = size(outputImage)
	outputImage = outputImage.astype(np.float)
	#Evidencia os pixels que faziam parte da cor Azul
	for i in range(0, height):
		for j in range(0,width):
			if binary_bluePixels.item(i,j) == 0.0:
				outputImage.itemset((i,j), outputImage.item(i,j) * 1.2)#Deixa mais claros os pixels que antigamente nao conpunham a cor azul
			else:
				outputImage.itemset((i,j), outputImage.item(i,j) * 0.8)#Deixa mais escuro os pixels que antigamente conpunham a cor azul

			#Trata a saturacao
			if outputImage.item(i,j) < 0.0:
				outputImage.itemset((i,j), 0.0)
			if outputImage.item(i,j) > 255.0:
				outputImage.itemset((i,j), 255.0)

	outputImage = outputImage.astype(np.uint8)
	cv2.imwrite("002.ImagemAposOProcessamentoDoHSV.png",outputImage)
	#cv2imshow(outputImage)
	return outputImage
"""
######################################################################################################################################
################## F I M   D A S   F U N C O E S   A U X I L I A R E S   D O   A L G O R I T I M O ###################################
######################################################################################################################################
##################            I N I C I O   D O   A L G O R I T I M O   P R O P O S T O             ##################################
######################################################################################################################################
"""

def WBC_SegProposed(imagepath, imshow = 0, saveResultFolder='Resultado', HSV_Processing = 0, imageName=''):
	#Exibe a imagem original
	orig = cv2.imread(imagepath,1)
	if imshow==1:
		cv2imshow(orig,windowName='Imagem Original')
	else:
		if not os.path.exists(saveResultFolder):
			os.makedirs(saveResultFolder)
		if not os.path.exists("00.ResultadosGerais"):
			os.makedirs("00.ResultadosGerais")

		cv2.imwrite(saveResultFolder+"/00.Imagem_Original.png",orig)
		cv2.imwrite("00.ResultadosGerais/"+imageName+"_Imagem Original.png",orig)

	#Passo 1 - Converter uma imagem de entrada A em uma imagem em escala de cinza
	if HSV_Processing:
		img = hsv_Processing(orig)
	else:
		img = imreadgray(imagepath)

	if imshow==1:
		cv2imshow(img,windowName='Passo 1 - Imagem em escala de cinza')
	else:
		cv2.imwrite(saveResultFolder+"/01.Imagem_Grayscale.png",img)

	#Passo 2 - Ajuste linear de contraste para obter a imagem 'L'
	L = linearContrastStretching(img)
	if imshow==1:
		cv2imshow(L, windowName='Passo 2 - Imagem em escala de cinza apos o ajuste linear de contraste')
	else:
		cv2.imwrite(saveResultFolder+"/02.L-Imagem_EscalaDeCinza_apos_ajuste_Linear_de_contraste.png",L)

	#Passo 3 - Equalizacao de histograma para obter a imagem 'H'
	H = histeq(img);
	if imshow==1:
		cv2imshow(H, windowName='Passo 3 - Imagem em escala de cinza apos a equalizacao de histograma')
	else:
		cv2.imwrite(saveResultFolder+"/03.H-Imagem_EscalaDeCinza_apos_equalizacao_de_histograma.png",H)

	#Passo 4 - Obter a imagem R1 = L + H
	R1 = imadd(L,H)
	if imshow==1:
		cv2imshow(R1, windowName='Passo 4 - R1 = L + H')
	else:
		cv2.imwrite(saveResultFolder+"/04.R1_L+H.png",R1)

	#Passo 5 - Obter a imagem R2 = L - H
	R2 = imsubtract(R1,H)
	if imshow==1:
		cv2imshow(R2, windowName='Passo 5 - R2 = L - H')
	else:
		cv2.imwrite(saveResultFolder+"/05.R2_L-H.png",R2)

	#Passo 6 - Obter a imagem R3 = R1 + R2
	R3 = imadd(R1,R2)
	if imshow==1:
		cv2imshow(R3, windowName='Passo 6 - R3 = R1 + R2')
	else:
		cv2.imwrite(saveResultFolder+"/06.R3_R1+R2.png",R3)

	#Passo 7 - Aplicar tres vezes um filtro de minima com a mascara 3x3 => [ [1,1,1]; [1,1,1]; [1,1,1] ]
	R3 = minimumfilter(R3)
	R3 = minimumfilter(R3)
	R3 = minimumfilter(R3)
	if imshow==1:
		cv2imshow(R3, windowName='Passo 7 - R3 apos o 3x filtro de minima')
	else:
		cv2.imwrite(saveResultFolder+"/07.R3_Apos_filtro_de_minima_tres_vezes.png",R3)

	#Passo 8 - Calcular o valor otimo de threshold usando o metodo de otsu
	#Passo 9 - Converter R3 para binaria usando o threshold calculado anteriormente
	binaryImage = otsuglobalthreshold(R3)
	if imshow==1:
		cv2imshow(binaryImage, windowName='Passo 8 e 9 - Imagem binaria feita com o threshold de otsu')
	else:
		cv2.imwrite(saveResultFolder+"/08_09.imagemBinaria_Apos_aplicar_threshold_de_otsu.png",binaryImage)

	#Passo extra - Inverter a imagem
	binaryImage = invertImage(binaryImage)
	if imshow==1:
		cv2imshow(binaryImage, windowName='Passo extra - Imagem binaria invertida')
	else:
		cv2.imwrite(saveResultFolder+"/09.1.ImageBinariaInvertida.png",binaryImage)

	#Passo 10 - Usar a abertura morfologica para remover pequenos grupos de pixels. Usar um elemento estruturante na forma de disco 9x9 
	binaryImage = morphologicalopening(binaryImage, mascara)
	if imshow==1:
		cv2imshow(binaryImage, windowName='Passo 10 - Imagem binaria apos abertura morfologica')
	else:
		cv2.imwrite(saveResultFolder+"/10.Imagem_binaria_apos_abertura_morfologica.png",binaryImage)

	#Passo 11 - Conectar os pixels vizinhos
	neighboringPixels = connectneighboringpixels(binaryImage)
	if imshow==1:
		cv2imshow(neighboringPixels, windowName='Passo 11 - Pixels vizinhos conectados')
	else:
		cv2.imwrite(saveResultFolder+"/11.Pixels_vizinhos_conectados.png",neighboringPixels)

	#Passo 12 - Remover os objetos com area menos do que a metade da area media de uma RBC
	finalAreas = removeIrrelevantObjects(binaryImage)
	if imshow==1:
		cv2imshow(finalAreas, windowName='Passo 12 - Eliminados os objetos com area menor do que metado de uma RBC')
	else:
		cv2.imwrite(saveResultFolder+"/12.Objetos_irrelevantes_eliminados.png",finalAreas)

	width, height = size(orig)
	for i in range(0, height):
		for j in range(0, width):
			if finalAreas.item(i,j) == 0:
				orig.itemset((i,j,0), 0)
				orig.itemset((i,j,1), 0)
				orig.itemset((i,j,2), 0)
	if imshow==1:
		cv2imshow(orig, windowName='Resultado final da segmentacao')
	else:
		cv2.imwrite(saveResultFolder+"/13.Resultado_final.png",orig)
		cv2.imwrite("00.ResultadosGerais/"+imageName+"_Imagem Segmentada.png",orig)








"""
		INICIO DA INTERFACE GRAFICA
"""
from Tkinter import *
import tkMessageBox
import tkFileDialog

class Packing:
	def __init__(self,instancia_Tk):

		self.container1 = Frame(instancia_Tk)
		self.container2 = Frame(instancia_Tk)
		self.container3 = Frame(instancia_Tk)
		self.container4 = Frame(instancia_Tk)

		#Botao que seleciona uma unicaImagem
		self.botaoSelecionaImagem = Button(self.container1,text='Selecionar uma imagem')
		self.botaoSelecionaImagem.bind("<Button-1>",self.selecionaImagem)
		self.botaoSelecionaImagem.pack()

		#Botao que seleciona multiplas imagens
		self.executarMultiplasImgs = Button(self.container1,text='Selecionar Multiplas imagens')
		self.executarMultiplasImgs.bind("<Button-1>",self.executarMultiplasImagens)
		self.executarMultiplasImgs.pack()

		#Label que serve para indicar uma imagem selecionada
		self.nomeImagemSelecionada = Label(self.container2)
		self.nomeImagemSelecionada.pack()
		
		#Botao para executar uma unica imagem passo a passo
		self.butaoPassoAPasso = Button(self.container3,text='Executar Passo a Passo')
		self.butaoPassoAPasso.bind("<Button-1>",self.executarPassoAPasso)
		
		#Botao para executar uma unica imagem em uma so passada
		self.butaoExecutarTotalmente = Button(self.container3,text='Executar Algoritimo')
		self.butaoExecutarTotalmente.bind("<Button-1>",self.executarTotalmente)

		#Label para indicar se o sistema esta processando alguma imagem
		self.estadoProcessamento = Label(self.container4)
		self.estadoProcessamento.pack()
		

		self.container1.pack()
		self.container2.pack()
		self.container3.pack()
		self.container4.pack()

		Canvas(instancia_Tk, width=500, height=50).pack()
		#instancia_Tk.resizable(width=False, height=False)

	#Quando o botao para selecionar uma unica imagem eh selecionado esta funcao eh executada
	def selecionaImagem(self, event):
		#Abre a caixa de dialogo para selecionar UMA imagem
		imageFile = tkFileDialog.askopenfile()
		#Extrai o nome do diretorio
		self.imagePath = imageFile.name
		#Atualiza o endereco da imagem na interface grafica
		self.nomeImagemSelecionada['text'] = self.imagePath

		self.iniciarAlg = Button(self.container3,text='Iniciar Algoritmo')

		#Exibe os botoes de opcao para o processamento da imagem selecionada
		self.butaoPassoAPasso.pack(side=RIGHT)
		self.butaoExecutarTotalmente.pack(side=RIGHT)

	#Quando o botao para executar passo a passo eh selecionado esta funcao eh executada
	def executarPassoAPasso(self, event):
		#Executa a funcao 'WBC_SegProposed' exibindo as imagens passo a passo
		WBC_SegProposed(self.imagePath, imshow = 1, HSV_Processing = 1)

	#Quando o botao executar algoritmo eh selecionado esta funcao eh executada
	def executarTotalmente(self, event):
		#Atualiza a interface grafica
		self.estadoProcessamento['text'] = "\nProcessando...\nAguarde"
		self.container4.update()

		#Executa a funcao 'WBC_SegProposed' salvando os resultados na pasta 'Resultados'
		WBC_SegProposed(self.imagePath, imshow = 0, HSV_Processing = 1)

		#Atualiza a interface grafica
		self.estadoProcessamento['text'] = ""
		tkMessageBox.showinfo("Fim do processamento", "Fim do processamento:\nCheque o resultado dentro da pasta 'Resultado'")

		#Exibi o resultado final do processamento
		cv2imshow(cv2.imread('Resultado/13.Resultado_final.png',1), windowName='Resultado final')


	#Quando multiplas imagens sao selecionadas essa funcao eh executada
	def executarMultiplasImagens(self, event):
		#Abre a caixa de dialogo para selecionar varias imagens. Eh retornado uma lista com os diretorios
		filesPaths = tkFileDialog.askopenfilenames()
		#Pega a quantidade de imagens selecionadas
		numImages = len(filesPaths)

		#Atualiza a interface grafica
		self.nomeImagemSelecionada['text'] = ""
		self.butaoPassoAPasso.pack_forget()
		self.butaoPassoAPasso.show = 0
		self.butaoExecutarTotalmente.pack_forget()
		self.butaoExecutarTotalmente.show = 0

		#Verifica se a pasta para salvar os resultados ja existe. Se nao, sao criadas
		if not os.path.exists("Resultado_Multiplas_Imagens"):
			os.makedirs("Resultado_Multiplas_Imagens")

		#Define o diretorio raiz para salvar os resultados
		rootDirectory = "Resultado_Multiplas_Imagens/"
		
		#Para cada imagem...
		for i in range(0, numImages):
			#Encontrar a posicao de todos os caracteres '/' que existem no endereco
			indexs = [j for j, letter in enumerate(filesPaths[i]) if letter == '/']

			#Extrair o nome da imagem do caminho do diretorio
			imageName = (filesPaths[i])[max(indexs)+1:len(filesPaths[i])-4]

			#Atualiza a interface grafica
			self.estadoProcessamento['text'] = "\nProcessando "+str(i+1)+" de "+str(numImages)+ ":\n"+imageName
			self.container4.update()
			
			#Constroi a string do endereco no qual os resultados seram salvos
			saveDirectory = rootDirectory+imageName

			#Executa o algoritmo  para a imagem e salva o resultado no diretorio especificado
			WBC_SegProposed(filesPaths[i], imshow = 0, saveResultFolder=saveDirectory, HSV_Processing = 1, imageName = imageName)

		#Atualiza a interface grafica
		self.estadoProcessamento['text'] = ""
		tkMessageBox.showinfo("Fim do processamento", "Fim do processamento:\nCheque o resultado dentro da pasta 'Resultado_Multiplas_Imagens'")

		


raiz=Tk()
Packing(raiz)
raiz.mainloop()

