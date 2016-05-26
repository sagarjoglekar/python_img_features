import numpy, Image
import colorsys
import os
import csv
from multiprocessing import Pool
import math

def rgb_to_hsl_hsv(a, isHSV=True):
    """
    Converts RGB image data to HSV or HSL.
    :param a: 3D array. Retval of numpy.asarray(Image.open(...), int)
    :param isHSV: True = HSV, False = HSL
    :return: H,S,L or H,S,V array
    """
    R, G, B = a.T

    m = numpy.min(a, 2).T
    M = numpy.max(a, 2).T

    C = M - m #chroma
    Cmsk = C != 0

    # Hue
    H = numpy.zeros(R.shape, int)
    mask = (M == R) & Cmsk
    H[mask] = numpy.mod(60 * (G[mask] - B[mask]) / C[mask], 360)
    mask = (M == G) & Cmsk
    H[mask] = (60 * (B[mask] - R[mask]) / C[mask] + 120)
    mask = (M == B) & Cmsk
    H[mask] = (60 * (R[mask] - G[mask]) / C[mask] + 240)
    H *= 255
    H /= 360 # if you prefer, leave as 0-360, but don't convert to uint8


    # Saturation
    S = numpy.zeros(R.shape, int)

    if isHSV:
        # This code is for HSV:
        # Value
        V = M

        # Saturation
        S[Cmsk] = ((255 * C[Cmsk]) / V[Cmsk])
        # H, S, and V are now defined as integers 0-255
        return H.swapaxes(0, 1), S.swapaxes(0, 1), V.swapaxes(0, 1)
    else:
        # This code is for HSL:
        # Value
        L = 0.5 * (M + m)

        # Saturation
        S[Cmsk] = ((C[Cmsk]) / (1 - numpy.absolute(2 * L[Cmsk]/255.0 - 1)))
        # H, S, and L are now defined as integers 0-255
        return H.swapaxes(0, 1), S.swapaxes(0, 1), L.swapaxes(0, 1)


def rgb_to_hsv(rgb):
    return rgb_to_hsl_hsv(rgb, True)


def rgb_to_hsl(rgb):
    return rgb_to_hsl_hsv(rgb, False)
	
def statistics(m):
	return [m.mean(), m.std(), m.max(), m.min()]
def brightness(l):
	return statistics(l)

def saturation(s):
	return statistics(s)

def colorfulness(r, g, b):
	rg = r - g
	yb = (r+g)/2 - b
	a = ((rg.std())**2)+((yb.std())**2) 
	b = ((rg.mean())**2)+((yb.mean())**2) 
	cf = numpy.sqrt(a) + 0.3 * numpy.sqrt(b)
	return cf
def contrast(l):
	return l.std()
def hue(h):
	return h.std()
def naturalness(H, L, S):
#--------
# K. Huang et al. Natural color image enhancement and evaluation algorithm based on human visual system. Computer Vision and Image Understanding,103(2006),52-63.
#--------
	skin_s = []
	grass_s = []
	sky_s = []
	for x in range(H.shape[0]):
		for y in range(H.shape[1]):
			l = L.item(x, y)
			h = H.item(x, y)
			s = S.item(x, y)/256.
			if l>80 or l<20 or s<=0.1:
				continue
			if h<70 and h>25:
				skin_s.append(s)
			elif h< 135 and h>95:
				grass_s.append(s)
			elif h< 260 and h>185:
				sky_s.append(s)
	if len(sky_s) + len(grass_s) + len(skin_s) ==0:
		return 0
	skin = numpy.array(skin_s)
	grass = numpy.array(grass_s)
	sky = numpy.array(sky_s)
	if len(skin_s) == 0:
		N_skin = 0
	else:
		N_skin = math.exp(-0.5*((skin.mean()-0.76)/0.52)**2)
	if len(grass_s) == 0:
		N_grass = 0
	else:
		N_grass = math.exp(-0.5*((grass.mean()-0.81)/0.53)**2)
	if len(sky_s) == 0:
		N_sky = 0
	else:
		N_sky = math.exp(-0.5*((sky.mean()-0.43)/0.22)**2)
	N = (N_skin*len(skin_s) + N_grass*len(grass_s) + N_sky*len(sky_s))/(len(skin_s)+len(grass_s)+len(sky_s))
	return N	


def image_process(img):
	img_dir = '/home/changtao/exp/image_analysis/img/'
	img_file = img_dir + img
	try:
		img_in = Image.open(img_file).convert('RGB')
	except Exception, e:
		print e
		return
	img_rgb = numpy.asarray(img_in, int)
	r, g, b = img_rgb.T
	h, l, s = rgb_to_hsl(img_rgb)
	return naturalness(h, l, s)
#	a = [img[:-4]] +brightness(l)+saturation(s) + [colorfulness(r, g, b), contrast(l)]
#	with open('data/color_features.csv', 'a') as f:
#		w = csv.writer(f, delimiter='|')
#		w.writerow(a)

def main():
	img_dir = '/home/changtao/exp/image_analysis/img/'
	file_list = os.listdir(img_dir)
	pool = Pool(20)
	n = 0
	input_list = []
	
	#---test--
#	for img in file_list:
#		n = image_process(img)
#		print img, n
#	return
	#---test --
	for img in file_list:
		n += 1
		input_list.append(img)
		if len(input_list)%1000 == 0:
			outp = pool.map(image_process, input_list)
			input_list = []
	if len(input_list) > 0:
		outp = pool.map(image_process, input_list)
		input_list = []
			
if __name__ == '__main__':
#	image_process('262897696969787600.jpg')
#	f = os.listdir('img/')
#	for i in f:
#		image_process(i)
	main()
