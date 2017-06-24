import re


def load_data(path_to_data):
    mrs_orig = []
    mrs_delex = []
    stack_sequences = []
    phrase_sequences = []

    with open(path_to_data) as f:
        for line in f:
            # original MR
            if re.match(r'FULL_DA', line):
                sep_equal = line.find('=')
                if sep_equal > -1:
                    mr = line[sep_equal + 1:].strip()
                    #mrs_orig.append(mr_to_dict(mr))
                    mrs_orig.append(mr_to_stacks(mr))
                else:
                    print('Error: failed to parse the MR in the following line:')
                    print(line)
            # delexicalized MR
            elif re.match(r'ABSTRACT_DA', line):
                sep_equal = line.find('=')
                if sep_equal > -1:
                    mr = line[sep_equal + 1:].strip()
                    #mrs_delex.append(mr_to_dict(mr))
                    mrs_delex.append(mr_to_stacks(mr))
                else:
                    print('Error: failed to parse the MR in the following line:')
                    print(line)
            # utterance
            elif re.match(r'->', line):
                # extract the annotated utterance from the line
                utterance = re.search(r'"(.*?)"', line)
                if utterance:
                    stacks, phrases = utterance_to_stacks(utterance.group(1))
                    stack_sequences.append(stacks)
                    phrase_sequences.append(phrases)
                else:
                    print('Error: failed to parse the utterance in the following line:')
                    print(line)

    return mrs_orig, mrs_delex, stack_sequences, phrase_sequences


def mr_to_dict(mr):	
    mr_dict = {}

    sep_left_paren = mr.find('(')
    sep_right_paren = mr.find(')')

    # parse the type of the dialogue act (DA)
    da_type = mr[:sep_left_paren].strip()
    
    # parse the slot-value pairs
    for slot_value in mr[sep_left_paren + 1:sep_right_paren].split(','):
        slot_and_value = slot_value.split('=')
        slot, value = slot_and_value[0].strip(), slot_and_value[1].strip().strip('"')
        if slot in mr_dict:
            mr_dict[slot].append(value)
        else:
            mr_dict[slot] = [value]
            
    return mr_dict


def mr_to_stacks(mr):	
    stacks = []

    sep_left_paren = mr.find('(')
    sep_right_paren = mr.find(')')

    # parse the type of the dialogue act (DA)
    da_type = mr[:sep_left_paren].strip()
    
    # parse the slot-value pairs
    for slot_value in mr[sep_left_paren + 1:sep_right_paren].split(','):
        slot_and_value = slot_value.split('=')
        slot, value = slot_and_value[0].strip(), slot_and_value[1].strip().strip('"')
        
        # ignore the "type" slot, as it always has a value "placetoeat"
        if slot == 'type':
            continue
        # ignore the "name" slots with the value "none"
        if slot == 'name' and value == 'none':
            continue
        # replace the value placeholders in the MR with "X"
        if re.match(r'X[0-9]', value):
            value = value[0]

        stacks.append([da_type, slot, value])
            
    return stacks


def utterance_to_stacks(utterance):
    stacks = []
    phrases = []
    
    # parse the (annotation, phrase)-pairs
    matches = re.findall(r'\[(.*?)\]\s*(.*?)\s*(?=\[|$)', utterance)

    for stack, phrase in matches:
        # DEBUG PRINT
        #print(stack, ':', phrase)

        if len(stack) > 0:
            stacks.append(['inform'] + stack.replace('_', ' ').split('+'))      # make the bottom value be 'inform'
        else:
            stacks.append(['inform'])

        phrases.append(phrase)
    
    return stacks, phrases
