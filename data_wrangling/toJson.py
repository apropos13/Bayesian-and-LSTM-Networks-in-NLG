from collections import deque
import re
import json
import random
import os, os.path, errno

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')



def da_to_dict(da):	
	d=dict()
	for item in da[17:-2].split(","): #split in comma
		#print item
		to_add=item.split("=")#split in =
		k,v=to_add[0],to_add[1] #take key value
		if k in d: #create dict with lists
			d[k].append(v)
		else:
			d[k]=[v]
	return d

def lexicalize(da, utt):
	d=da_to_dict(da)
	new_utt=""
	occ=1 #to mark every second occurance of a variable X
	for index in range(0,len(utt)):
		if utt[index]=='X' and occ==2: #if a variable name is found
			occ=1 #reset
			#backtrack to find starting [
			match_c=""
			match_i=index
			

			#get beginning of placeholder (starting after [ )
			while match_c!="[":
				match_i=match_i-1
				match_c = utt[match_i]

			

			placeholder=""
			#get placeholder name (go up to +)
			while utt[match_i+1]!="+":
				#print "utt=",utt[match_i+1]
				placeholder=placeholder+utt[match_i+1]
				match_i=match_i+1

			
			value=d[placeholder]
			new_utt=new_utt+value[0]

			value.pop(0)
			#print value
			d[placeholder]= value #update list of elements since element was just used


		else:
			if utt[index]=='X':
				occ+=1
			new_utt=new_utt+utt[index]

	return new_utt



def utt_final_json(utt):
	#remove everything with parens
	new_utt=""
	switch=True
	for index in range(0, len(utt)):
		if utt[index]=='[':	
			switch=False

		if switch:
			new_utt=new_utt+utt[index]

		if utt[index]==']':
			switch=True

		
	new_utt=new_utt.replace('"','')
	new_utt=new_utt.replace(';','')
	return new_utt[3:] #discard the arrow

def da_final_json(da):
	new_da=da.replace(',',';')
	return new_da[10:]

def finalize(split_point):

	with open("ACL10-inform-training.txt") as f:
		final_list=[]
		temp_list=[]
		da_tracker=""
		for line in f:
			if line[0]=='F': #start with an F
				temp_list.append(da_final_json(line))
				da_tracker=line
				
				
			elif line[0]=='-':
				temp_list.append(utt_final_json(lexicalize(da_tracker, line)))
				final_list.append(temp_list)
				temp_list=[]

	#30% train, rest test and validation
	print("permuting data. creating train and validation sets...")
	random.shuffle(final_list)
	n_points=len(final_list)
	break_point=int(split_point*n_points)
	train_set=final_list[: break_point]
	test_set=final_list[break_point: ]
	path_train='../RNNLG/data/original/n2e/Split%.2f/'%(split_point)

	with safe_open_w(path_train+'train.json') as f:
		json.dump(train_set, f, indent=4, ensure_ascii=False)

	with safe_open_w(path_train+'test.json') as f:
		json.dump(test_set, f, indent=4, ensure_ascii=False)

	with safe_open_w(path_train+'valid.json') as f:
		json.dump(test_set, f, indent=4, ensure_ascii=False)



if __name__=="__main__":
	sample_da='FULL_DA = inform(name="The Golden Curry",type=placetoeat,eattype=restaurant,near="The Six Bells in High Saint",area="romsey",food=Indian)'
	da_to_dict(sample_da)
	sample_utt='-> "[name+X]X []is an [food+Indian]Indian [eattype+restaurant]restaurant [area]in [area+X]X [near]near [near+X]X";'
	#lexicalize(sample_da ,sample_utt)
	#utt_final_json(lexicalize(sample_da,sample_utt))
	#da_final_json(sample_da)
	#finalize()

	split_points=[0.4,0.5,0.6,0.7,0.8,0.9,0.95]
	for s in split_points:
		finalize(s)

	

