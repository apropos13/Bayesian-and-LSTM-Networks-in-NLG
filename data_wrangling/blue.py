import subprocess
import sys
import ast
import re
import nltk
import os

def get_meteor(filename_utt,filename_pred):
	
	#add the path relative to the file that is going to run it!
	cmd='java -Xmx2G -jar ../data_wrangling/meteor-*.jar '+filename_utt+' '+filename_pred+ ' -norm -writeAlignments -f system1 -q'
	devnull = open(os.devnull, 'wb') #python >= 2.4
	meteor_score = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, stderr=devnull, universal_newlines=True).communicate()[0].strip('\n')
	return float(meteor_score)
	#print meteor_score


    
def get_blue(filename_utt,filename_pred):
	'''
	cmd='perl ../data_wrangling/good_blue.perl '+filename_utt+' < '+ filename_pred
	perl_bleu4 = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE).communicate()[0].split()[2].split('/')[3]
	return float(perl_bleu4)
	#to get bleu 1,2,3 simply change [3]-->[1] or [2] or [3]
	'''

	utterance_list=[]
	with open(filename_utt) as data_file:
		for line in data_file:
			utterance_list.append(line.split())

	#print utterance_list[1]
	#print len(utterance_list)

	prediction_list=[]
	with open(filename_pred) as pred_file:
		for line in pred_file:
				prediction_list.append(line.split())

	#print prediction_list[1]
	#print len(prediction_list)

	#sanity check
	assert len(prediction_list)==len(utterance_list), "Number of predictions is not equal to the number of refernces"

	#print("Getting Blue corpus score...")
	BLEU_coprus= nltk.translate.bleu_score.corpus_bleu(utterance_list, prediction_list) 
	return BLEU_coprus

