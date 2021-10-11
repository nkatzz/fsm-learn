import csv


# dividing in chunks of size = chunk_size, with the most equalized number of positive and negative examples as it can be
# f.e return_chunks(100, training_dataset_path) => returns: a list of batches with len = floor(|dataset_num|/a*) ; chunk_list[i]: i'th batch
# a := the number of cell types that exist in the dataset, f.e {alive, apoptotic, necrotic} => a=3
def return_chunks(chunk_size, training_path, a=3):    
    
    # data := dataset
    data = []
    csvReader = csv.reader(open(training_path, newline='\n'))
    for row in csvReader:
        row = ",".join(row)
        data.append(row)
    
    # positive examples
    positive = [x for x in data if x[-3]=='1']
    # negative examples
    negative  = [x for x in data if x[-3]=='0']
    
    # defining the majority of target of the examples (Etc negative examples) and their difference
    difference=abs(len(positive)-len(negative))
    min_len=len(positive)>len(negative) and len(negative) or len(positive)
    max_len= min_len==len(positive) and len(negative) or len(positive)
    is_positive_less=len(positive)>len(negative) and False or True
    
    # data multiplied with a because with have got a types of cell
    # all data = #chunks*a*chunk_size
    chunks=len(data)//(a*chunk_size)
    chunks_list = [[] for i in range(chunks)]
    # chunk_step: max data that the dataset with less elements can have distributed along the chunks
    chunk_step=min_len//(a*chunks)
    # diff_step: extra data that the dataset with the majority of elements has got
    # chunk_step + chunk_step + diff_step = chunk_size
    diff_step=chunk_size-2*chunk_step
    
    # constructing the batches, using the parameters defined above
    # diff_step: again multiplying with a, because it is defined using chunk_size and chunk_step which are not multiplied with a
    start,start2,end,end2=0,0,a*chunk_step,a*(chunk_step+diff_step)
    for i in range(chunks):        
        if(is_positive_less):
            chunks_list[i]+=positive[start:end]
            chunks_list[i]+=negative[start2 :end2]
        else:
            chunks_list[i]+=negative[start:end]
            chunks_list[i]+=positive[start2 :end2]
        start+=a*chunk_step
        start2+=a*(chunk_step+diff_step)
        if(i!= (chunks-2)):
            end+= a*chunk_step
            end2+= a*(chunk_step+diff_step)
        else:
            end=min_len
            # here we are dropping examples, if there are too much from the "majority target"
            # !!!! must be determined !!!! 
            end2=min(end2+a*(chunk_step+diff_step),max_len)
        print("positive: ", end-start, "negative: ", end2-start2)
    return chunks_list


# insert one type of cell's data; returns a slice of the data, until the time t
# it's a dummy way...
# to be used from get_dataset_sliced_until_time_t below  
def get_data_until_time_t(data,t):
    cnt=0
    new_data=''
    # when we find the i'th '.', we have got data's info until time i
    for i in range (len(data)):
        if data[i] == '.':
            cnt+=1
            if(cnt==t):
                new_data+='.'
                break
        new_data+=data[i]

    # appending data's class
    tmp=''
    for i in range(len(data)-1,0,-1):
        tmp+=data[i]
        if (data[i]=='c'):
            break
    return new_data + ' '+ tmp[::-1]

# insert a dataset as a list; returns all the data until time t
# f.e  get_dataset_sliced_until_time_t(dataset: list, t: integer)
# where you can obtain dataset as depicted below: training_path: train -> list: dataset
"""
dataset = []
csvReader = csv.reader(open(train, newline='\n'))
for row in csvReader:
    row = ",".join(row)
    dataset.append(row)
"""
def get_dataset_sliced_until_time_t(dataset,t):
    new_data=[]
    for i in range (len(dataset)):
        new_data.append(get_data_until_time_t(dataset[i],t))
    return new_data


# including only a subset of a specific type of cells (f.e. only necrotic cells)
# include_list: include the type of cells you would like to be included (alive, apoptotic, necrotic)
def mts_subsets(train,test, include_list):
    
    if (len(include_list)<1 or len(include_list)>2):
        print("Oopsie...")
    
    if (len(include_list)==1):
        if('alive' in include_list):
            div,ro=1,6
        elif('apoptotic' in include_list):
            div,ro=2,10
        else:
            div,ro=0,9
    else:
        if('alive' not in include_list):
            div,ro=1,6
        elif('apoptotic' not in include_list):
            div,ro=2,10
        else:
            div,ro=0,9
        
    
    
    cnt=1
    with open(train, "r") as my_input_file1, open("train.csv", 'w',newline='') as myfile:
        for row in csv.reader(my_input_file1):
            if (len(include_list)==1 and cnt%3==div) or (len(include_list)==2 and (not cnt%3==div)):
                csv.writer(myfile).writerow(row)
            cnt+=1

    cnt=1
    with open(test, "r") as my_input_file1, open("test.csv", 'w', newline='') as myfile:
        for row in csv.reader(my_input_file1):
            if (len(include_list)==1 and cnt%3==div) or (len(include_list)==2 and (not cnt%3==div)):
                csv.writer(myfile).writerow(row)
            cnt+=1


