import os
import matplotlib.pyplot as plt
import numpy as np

def get_counts(directory):
	distribution_container=[] #get empty number of utterances for every different split
	for file in os.listdir(directory):
		counter=0
		if file.endswith(".txt"):
			path_to_file=directory+file
			with open(path_to_file, mode='r') as opened:
				for line in opened:
					if line[0]=='R' and line[5]=='[' and line[6]==']':
						counter+=1
				distribution_container.append(counter)

			#print distribution_container

	total=sum(distribution_container)
	return total, distribution_container


if __name__ == '__main__':
	b0, b0_distr=get_counts('../bagel/results/Backoff0/')
	b1, b1_distr=get_counts('../bagel/results/Backoff1/')
	b2, b2_distr=get_counts('../bagel/results/Backoff2/')
	b3, b3_distr=get_counts('../bagel/results/Backoff3/')
	b4, b4_distr=get_counts('../bagel/results/Backoff4/')

	print("b0="+str(b0))
	print(b0_distr)
	print("b1="+str(b1))
	print(b1_distr)
	print("b2="+str(b2))
	print("b3="+str(b3))
	print("b4="+str(b4))


	bins=[b0, b1, b2, b3, b4]

	ind=np.arange(len(bins)) #x location of data
	fig, ax = plt.subplots()
	my_width=0.35
	rects1 = ax.bar(ind, bins, width=my_width, color='b')

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Number of Empty Utterances', fontsize=16)
	#ax.set_title('Back-off Level')
	ax.set_xticks(0.0011+ ind + my_width / 2)
	ax.set_xticklabels(('Back-off 0', 'Back-off 1', 'Back-off 2', 'Back-off 3', 'Back-off 4'))
	ax.yaxis.set_label_coords(-.08, 0.5)

	for label in ax.xaxis.get_ticklabels():
	    # label is a Text instance
	    label.set_color('blue')
	    label.set_rotation(25)
	    label.set_fontsize(16)


	plt.show()


