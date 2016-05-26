from color import brightness, saturation, colorfulness, contrast, naturalness, rgb_to_hsl
from multiprocessing import Pool
import os
import numpy, Image
import csv
import math, colorsys

out_file = '/home/changtao/exp/image_analysis/data/features.csv'
img_dir = '/home/changtao/data/pnt_img/filtered/'
def image_process(img):
	img_file = img_dir + img
	try:
		img_in = Image.open(img_file).convert('RGB')
	except Exception, e:
		print e
		return
	img_rgb = numpy.asarray(img_in, int)
	r, g, b = img_rgb.T
	h, l, s = rgb_to_hsl(img_rgb)
#	return naturalness(h, l, s)
	a = [img[:-4]] +brightness(l)+saturation(s) + [colorfulness(r, g, b), contrast(l), naturalness(h, l, s)]
	with open(out_file, 'a') as f:
		w = csv.writer(f, delimiter='|')
		w.writerow(a)

def test():
	file_list = os.listdir(img_dir)
	
	for img in file_list:
		n = image_process(img)
		print img, n
	return


def main():
	file_list = os.listdir(img_dir)
	pool = Pool(10)
	n = 0
	input_list = []
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
	main()
#	test()
