import clingo
import multiprocessing
import time
import csv
import os
from utilities import *


# call solver for the whole dataset
# parameters are the same as defined on class app; some useful notes explained below:
# train: if incremental_mode== True => must be a string, else => must be a list
# test: must be a string
def call_solver(train, test, states,  target_class, exact = False, incremental_mode= False, t_slice=0, timeLimit=float('inf')):
    model_data={}
    if (incremental_mode):   
        app = App(train, test, states,  target_class, exact, incremental_mode, t_slice, time=timeLimit)
    else:
        app = App([train], test, states,  target_class, exact, time=timeLimit)
    
    # computing metrics
    cnt_time=time.time()
    app.induce()
    model_data['time']=time.time()-cnt_time
    model_data['best_model']=app.best_model
    model_data['train_vector']=app.training_vector
    model_data['test_score']=app.best_test_score
    model_data['test_vector']=app.test_vector
    model_data['test_f1_accuracy']=app.f1_accuracy
    model_data['test_earliness']=app.test_earliness
    
    # computing metrics for train set (test set = train set)
    saved_model, saved_model_size = app.best_model, app.model_size
    app = App(train, train, states,  target_class, exact, incremental_mode, t_slice, timeLimit,  only_test=True, saved_model= saved_model, saved_model_size=saved_model_size)
    app.induce()
    model_data['train_score']=app.best_test_score
    model_data['train_f1_accuracy']=app.f1_accuracy
    model_data['train_earliness']=app.test_earliness
    
    if (incremental_mode):
        os.remove('slicedDataset.csv')
    return model_data

# 2 algorithms used; 
# first one "incremental searching": whenever test score > test threshold => add current batch (and call the solver again)
# second one "incremental_all searching": add current batch anyway and only call  the solver when: test score > test threshold, using all the dataset seen until this time
# extra parameters, compared with those defined on class app, to be defined: 
# chunk size: the number of the examples included into each batch (for more info: utilities/return_chunks)
# test_threshold: in each step, namely when proceeding to the next batch, checking if current_models(test_score) <= test_threshold in order to ignore this batch; or else => revision (depending on the type of algorithm used) 
def call_solver_incremental_search(chunk_size, test_threshold, train, test, states,  target_class, exact = False, incremental_mode= False, t_slice = 0, timeLimit=float('inf')):

    # training dataset -> list of chunks
    chunks_list=return_chunks(chunk_size, train)
    batch_len=len(chunks_list[0])
    # first batch initiliazed
    with open("currentBatch.csv", 'w') as myfile:
        wr = csv.writer(myfile,delimiter="\n")
        wr.writerow(chunks_list[0])
    with open("allBatches.csv", 'w') as myfile:
        wr = csv.writer(myfile,delimiter="\n")
        wr.writerow(chunks_list[0])
    
    
    # defining parameters to keep in track
    # added_batch[i]= True => batch i+1 is conluded to the dataset (namely model isn't efficient enough when excluding it)
    added_batch=[False for i in range(len(chunks_list))]
    added_batch[0]=True

    # train_batch[i]= True => in batch i+1 incremental_all model is doing a revision
    train_batch=[False for i in range(len(chunks_list))]
    train_batch[0]=True
    # elapsed time := testing time when solver won't be called for revision  or else (testing + training + testing) time when we call solver again for revision
    # notation about the variables stored: bathes_{parameter}: "incremental searching", allBatches_{parameter}: "incremental_all searching" 
    
    batches_time, allBatches_time = 0, 0
    batches_test_score=[0.0 for i in range(len(chunks_list)) ]
    allBatches_test_score=[0.0 for i in range(len(chunks_list)) ]
    
    
    # ----------- incremental searching just started -----------------
    if (incremental_mode):   
        app = App("currentBatch.csv", "currentBatch.csv", states,  target_class, exact, incremental_mode, t_slice, time=timeLimit)
    else:
        app = App(["currentBatch.csv"], "currentBatch.csv", states,  target_class, exact, time=timeLimit)

    start_time = time.time()
    app.induce()
    allBatches_time=batches_time=time.time() - start_time
    allBatches_test_score[0]=batches_test_score[0]=app.best_test_score
    # storing current best model and it's size
    best_model = allBatches_best_model = app.best_model
    cur_size=allBatches_cur_size=app.model_size

    remains=len(chunks_list)-1  
    first_time = True
    cnt=1

    # accepted batch := when current best model's test score including this batch is greater than test threshold  
    # currentBatch := all accepted batches
    for batch in chunks_list:
        # batch[0] is already processed
        if first_time:
            first_time = False
            continue
        print(" Remain "+ str(remains)+ " batches")
        remains-=1
        # appending the current batch in allBatches.csv
        with open("allBatches.csv", 'a+') as myfile:
            wr = csv.writer(myfile,delimiter="\n")
            wr.writerow(batch)
        # current batch   
        with open("batch.csv", 'w') as myfile:
            wr = csv.writer(myfile,delimiter="\n")
            wr.writerow(batch)
        
        # "incremental_all searching"
        # test set := "batch.csv" = current batch
        app = App("allBatches.csv", "batch.csv", states,  target_class, exact, incremental_mode,  t_slice, time=timeLimit, only_test=True, saved_model=allBatches_best_model, saved_model_size=allBatches_cur_size)
        start_time = time.time()
        app.induce()
        whole_time=time.time()-start_time        

        # if this model is good enough, namely test score is less than a predefined threshold, we ignore this batch; else call the solver again, including this batch on the train set
        if app.test_score > test_threshold:
            print("This model isn't good enough (incremental_all) => revision is going to occur ", app.test_score)
            train_batch[cnt] = True
            if (incremental_mode):   
                app = App("allBatches.csv", "batch.csv", states,  target_class, exact, incremental_mode, t_slice, time=timeLimit)
            else:
                app = App(["allBatches.csv"], "batch.csv", states,  target_class, exact, time=timeLimit)
        
            start_time = time.time()
            app.induce()
            whole_time+=time.time() - start_time
            allBatches_best_model = app.best_model
            allBatches_cur_size=app.model_size
            
        
        allBatches_time+=whole_time
        allBatches_test_score[cnt]=app.test_score
        
        # "incremental searching"
        # only_test=True => current best model will be tested on test set; test set = current batch
        app = App("currentBatch.csv", "batch.csv", states, target_class, exact, incremental_mode, time=timeLimit, only_test=True, saved_model=best_model, saved_model_size=cur_size)
        start_time = time.time()
        app.induce()
        test_time=time.time()-start_time
        # if this model is good enough, namely test score is less than a predefined threshold, we ignore this batch
        if app.test_score <= test_threshold:
            batches_time+=test_time
            batches_test_score[cnt]=app.test_score
            cnt+=1
            continue
        print("This model isn't good enough (incremental) => revision is going to occur ", app.test_score)
        # else: add current batch, train the model and find the model that mininizes training error in this new dataset
        with open("currentBatch.csv", 'a+') as myfile:
            wr = csv.writer(myfile,delimiter="\n")
            wr.writerow(batch)
        
   
        added_batch[cnt]=True
        batch_len+=len(batch)        
        if (incremental_mode):   
            app = App("currentBatch.csv", "batch.csv", states,  target_class, exact, incremental_mode, t_slice, time=timeLimit)
        else:
            app = App(["currentBatch.csv"], "batch.csv", states,  target_class, exact, time=timeLimit)
        
        start_time = time.time()
        app.induce()
        batches_time+=(time.time() - start_time+test_time)
        batches_test_score[cnt]=app.best_test_score
        best_model = app.best_model
        cur_size=app.model_size
        cnt+=1
    
    # bookkeeping
    allBatches_model={}
    model={}
    model['best_model']=best_model
    allBatches_model['best_model']=allBatches_best_model
    # test set := train set => deriving info about the train set
    app = App("allBatches.csv", train, states,  target_class, exact, incremental_mode,  t_slice, time=timeLimit, only_test=True, saved_model=allBatches_best_model, saved_model_size=allBatches_cur_size)
    app.induce()
    allBatches_model['train_vector']=app.test_vector
    allBatches_model['train_score']=app.best_test_score
    allBatches_model['train_f1_accuracy']=app.f1_accuracy
    allBatches_model['train_earliness']=app.test_earliness
    allBatches_model['time']=allBatches_time
    app = App("currentBatch.csv", train, states, target_class, exact, incremental_mode, time=timeLimit, only_test=True, saved_model=best_model, saved_model_size=cur_size)
    app.induce()
    model['train_vector']=app.test_vector
    model['train_score']=app.best_test_score
    model['train_f1_accuracy']=app.f1_accuracy
    model['train_earliness']=app.test_earliness
    model['time']=batches_time
    model['added_batches']=added_batch
    model['batch_len']=batch_len

    # info about the test set
    app = App("allBatches.csv", test, states,  target_class, exact, incremental_mode,  t_slice, time=timeLimit, only_test=True, saved_model=allBatches_best_model, saved_model_size=allBatches_cur_size)
    app.induce()
    allBatches_model['test_score']=app.best_test_score
    allBatches_model['test_vector']=app.test_vector
    allBatches_model['test_f1_accuracy']=app.f1_accuracy
    allBatches_model['test_earliness']=app.test_earliness
    allBatches_model['train_batch']=train_batch
    
    app = App("currentBatch.csv", test, states, target_class, exact, incremental_mode, time=timeLimit, only_test=True, saved_model=best_model, saved_model_size=cur_size)
    app.induce()
    model['test_score']=app.best_test_score
    model['test_vector']=app.test_vector
    model['test_f1_accuracy']=app.f1_accuracy
    model['test_earliness']=app.test_earliness
    os.remove("currentBatch.csv")
    os.remove("batch.csv")
    os.remove("allBatches.csv")
    if (incremental_mode):
        os.remove('slicedDataset.csv')
    return model, allBatches_model


# run {call_solver, call_solver_incremental_search} for multiple number of states and different timeLimits used respectively
# parameters which are used differently than the cases in the functions defined above:
# states: list of states f.e. [1,2,3,4,5], timeLimits: list of timelimit for each state 
# incremental_search == True => call_solver_incremental_search is called, else call_solver is called
# analytics == True => complete runs for all states in order to keep analytics; or else just proceed to the next state only only if the derived model gets better         
def call_app_all_parameters(train, test, all_states,  timeLimits, target_class, exact = False, incremental_mode= False, t_slice=0, incremental_search = False, chunk_size=0, test_threshold=0, analytics = False):
    
    if (analytics):
        file1 = open("all_parameters_stats.txt","w")
    best_states, best_score =  0, float('inf')
    for (states,timeLimit) in zip(all_states,timeLimits):
        print(" ----- State " + str(states) + " -----")
        if (incremental_search):
            app,all_batches_app = call_solver_incremental_search(chunk_size, test_threshold, train, test, states,  target_class, exact, incremental_mode, t_slice, timeLimit=timeLimit)
        else:
            app = call_solver(train, test, states,  target_class, exact = exact, incremental_mode= incremental_mode, t_slice=t_slice, timeLimit=timeLimit)
        
        # storing the model with minimized training accuracy
        if (not analytics) and (app['train_score'] < best_score):
            best_app=app
            best_states=states
            if incremental_search:
                best_all_batches_app = all_batches_app

        # anylytics == False && app.best_training_acc >= best_score
        elif (not analytics):
            break
        # else just keep analytics
        else:
            file1.write("\n ----- States = " + str(states) + " -----\n")
            file1.write(app['best_model']+ "\n")
            file1.write(" Test vector: "+ str(app['test_vector'])+" , train vector:  " + str(app['train_vector']) + " , time elapsed: " + str(app['time'])+"\n")
    
    if incremental_search:
        return best_app,best_all_batches_app, best_states
    else:
        return best_app, best_states
    
    
    
# only for experimental purposes: train slicing, while the proccess doesn't take place into the class app but before it's called
# train, test: must be strings
def learning_with_sliced_data(time_slices, train,test,states, exact, target_class, timeLimit):
    
    # importing dataset from training path
    data = []
    csvReader = csv.reader(open(train, newline='\n'))
    for row in csvReader:
        row = ",".join(row)
        data.append(row)

    earliness_score,incremental_mode = float('inf'), False

    file1 = open("time_slices_experimenting.txt","w")
    file1.write("Time slices: [")
    
    first= True
    for elem in time_slices:
        if not first:
            file1.write(", ")
        else:
            first= False
        file1.write(str(elem))
    file1.write("]\n")
    file1.write("Earliness at stages: [")
    for t_slice in time_slices:
        # get dataset sliced until time t
        print(t_slice)
        cur_dataset=get_dataset_sliced_until_time_t(data,t_slice)
        with open("currentDataset.csv", 'w') as myfile:
            wr = csv.writer(myfile,delimiter="\n")
            wr.writerow(cur_dataset)
        # test set = training_set in order to choose the model on train set but also get the metrics desired while testing on train set
        app = App(["currentDataset.csv"], "currentDataset.csv", states, target_class, exact,  incremental_mode, time=timeLimit, only_test=False)
        app.induce()
        file1.write(str(app.test_earliness) + ", ")
        # if test_earliness doesn't improve, namely doesn't get decreased, stop looping
        if (earliness_score <= app.test_earliness):
            break
        # else update current's model info
        cur_model=app.best_model
        earliness_score=app.test_earliness
        cur_size=app.model_size
        cur_slice=t_slice
        train_vector=app.test_vector
        train_f1=app.f1_accuracy
        app = App(["currentDataset.csv"], "currentDataset.csv", states, target_class, exact, incremental_mode, time=timeLimit, only_test=True, saved_model=cur_model, saved_model_size=cur_size)
        app.induce()
    
    file1.write("]\n")
    file1.write("Searching terminated at slice t=" + str(cur_slice) + ", found: \n ")
    file1.write(str(cur_model))
    file1.write("\nStats on train set: error vector = " +str(train_vector)+", f_1 accuracy = "+ str(train_f1)+", earliness = " + str(earliness_score))
    # testing the best_model on the test set (test_only= True)
    app = App(["currentDataset.csv"], test, states, target_class, exact, incremental_mode, time=timeLimit, only_test=True, saved_model=cur_model, saved_model_size=cur_size)
    app.induce()
    #print("test on test set ", app.test_earliness,app.f1_accuracy)
    file1.write("\nStats on test set: error vector = " +str(app.test_vector)+", f_1 accuracy = "+ str(app.f1_accuracy)+", earliness = " + str(app.test_earliness))
    os.remove("currentDataset.csv")
    return cur_model

def show_model(model,best=True):
    if(best):
        print("Best Model: ")
    print(model)

"""
useful notes and tips about class App
1. metrics
train set: self.best_model, self.training_vector=[fns, fps, model.cost], self.model_size 
test set: self.test_vector=[fns, fps, model.cost], self.best_test_score= fns+fps, self.f1_accuracy, self.test_earliness

2. self.only_test == True
why/what: just test a specific model (do not use solver, searching for stable solutions)
parameters that must be passed: self.only_test = True, saved_model := model to be tested, saved_model_size := size of the model to be tested
b. useful idea: 
in order to obtain the metrics existing for the test set, as explained above, and for the training set you can use: test set= training set
f.e. 
app = App([training_path], training_path, states, exact, target_class, incremental_mode, time=timeLimit, only_test=True, saved_model=some_previous_best_model, saved_model_size=size_of_the_saved_model)
app.induce() 

3. incremental_mode == True
parameters that must be passed: t_slice when cell datas will be terminated
CAUTION: if incremental_mode == True => self.train must be str, else =>  self.train must be list
"""   

class App:


    best_training_acc=float('inf')
    # vectors of type: [#false negative, #false positive, model size]
    training_vector=[]
    test_vector=[]
    model_size=0
    current_model = ''
    best_model = ''
    test_score= 0.0
    best_test_score= 0.0
    f1_accuracy = 0.0
    test_earliness = 0.0
    model_couner = 1
    cores = multiprocessing.cpu_count()
    ctl = clingo.Control()

    def __init__(self, training_path, testing_path, states, target_class,  exact = False, incremental_mode=False, t_slice=0, time=float('inf'),only_test= False,saved_model='',saved_model_size=0):
        self.train = training_path
        self.test = testing_path
        self.target_class = target_class
        self.states = states
        self.exact = exact
        # train slicing algorithm
        self.incremental_mode = incremental_mode
        # for data in train set: data := data [:t_slice]
        self.t_slice=t_slice
        # defining timelimit
        self.time=time
        # defining when conducting incremental/batched learning, meaning "only test this model, don't call the solver"
        self.only_test=only_test
        # passing a previous model, in order only to test it on this run 
        self.saved_model=saved_model
        # passing the size of a previous model when trained, in order only to test it on this run 
        self.saved_model_size=saved_model_size


    def on_model(self, model):
        
        self.current_model = ' '.join(map(lambda x: x + '.', format(model).split(' ')))
        if self.current_model=='.':
            return
        print("\nModel {0} ".format(self.model_couner))

        # training accuracy: heuristic for searching
        accuracy_metric=model.cost[0]+model.cost[1]+model.cost[2]
        # keeping best model's info so far (even if timeLimit is defined) ; namely the model that minimizes training error
        # correctness: every time a model with better heuristic accuracy (accuracy_metric) is found => it's details are saved
        if  accuracy_metric < self.best_training_acc:
            self.best_training_acc = accuracy_metric
            self.best_model = self.current_model
            self.training_vector=model.cost
            self.model_size=model.cost[2]        

        self.model_couner += 1
        self.test_model()
        

    def show_test_results(self, test_results):

        def process_atom(atom, what):
            seqId = str(atom.arguments[0].string)
            prediction_point = int(atom.arguments[1].number)
            seq_length = int(atom.arguments[2].number)
            earlines = prediction_point/float(seq_length)
            if what == "tp":
                tp_seq_ids.append(seqId)
                tp_earliness_vals.append(earlines)
            else:
                fp_seq_ids.append(seqId)
                fp_earliness_vals.append(earlines)
        
        
        fns = 0
        tp_seq_ids, fp_seq_ids = [], []
        tp_earliness_vals, fp_earliness_vals = [], []
        for atom in test_results.symbols(atoms=True):
            if atom.match("tp", 3):
                process_atom(atom, "tp")
            elif atom.match("fp", 3):
                process_atom(atom, "fp")
            elif atom.match("fns", 1):
                fns = int(atom.arguments[0].number)
            else:
                pass
        
        tps = len(tp_seq_ids)
        fps = len(fp_seq_ids)
        precision = float(tps) / (tps + fps) if tps > 0 else 0.0
        recall = float(tps) / (tps + fns) if tps > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if tps > 0 else 0.0
        tp_earliness = float(sum(tp_earliness_vals)) / len(tp_earliness_vals) if len(tp_earliness_vals) > 0 else 'None'
        fp_earliness = float(sum(fp_earliness_vals)) / len(fp_earliness_vals) if len(fp_earliness_vals) > 0 else 'None'
        appended = tp_earliness_vals + fp_earliness_vals
        earliness = float(sum(appended)) / len(appended) if len(appended) > 0 else 'None'
        harmonic_mean = 2 * f1 * earliness / (f1 + earliness) if earliness != 'None' else 'None'
        
        if self.only_test:
            self.test_vector=[fns,fps,self.saved_model_size]
            self.test_score = fns+fps
            self.best_test_score=self.test_score
            self.f1_accuracy= f1
            self.test_earliness=earliness
            return

        # best model's information saved 
        if self.best_model == self.current_model:
            self.test_score = fns+fps
            self.best_test_score=self.test_score
            self.test_vector=[fns,fps,self.model_size]
            self.f1_accuracy= f1
            self.test_earliness=earliness

        """  
        print('Performance (testing set):\nTPs: {0}, FPs: {1}, FNs: {2}, F1: {3}\nTP earlines: {4}, FP earliness: {5}\nEarliness: {6}, '
              'F1/earliness harmonic mean: {7}'.format(tps, fps, fns, f1, tp_earliness, fp_earliness, earliness,
        
                                               harmonic_mean))
    
        """
        
    
    def induce(self):
        """
        For the solver configuration:
        >>> prg.configuration.keys
        ['tester', 'solve', 'asp', 'solver', 'configuration', 'share', 'learn_explicit', 'sat_prepro', 'stats', 'parse_ext', 'parse_maxsat']
        >>> prg.configuration.solve.keys
        ['solve_limit', 'parallel_mode', 'global_restarts', 'distribute', 'integrate', 'enum_mode', 'project', 'models', 'opt_mode']
        >>> prg.configuration.solve.opt_mode
        'opt'
        """
        # if only_test == true => just test the model on the test set
        # f.e while conducting incremental searching, we just need to test the model 
        if self.only_test:
            self.current_model=self.saved_model
            self.test_model()
            return

        # train slicing := all examples included into the train set will be considered until time self.t_slice   
        if self.incremental_mode:
            data = []
            csvReader = csv.reader(open(self.train, newline='\n'))
            for row in csvReader:
                row = ",".join(row)
                data.append(row)
            cur_dataset=get_dataset_sliced_until_time_t(data,self.t_slice)
            with open("slicedDataset.csv", 'w') as myfile:
                wr = csv.writer(myfile,delimiter="\n")
                wr.writerow(cur_dataset)
            self.train=["slicedDataset.csv"]
        
        ctl = clingo.Control()
        ctl.configuration.solve.parallel_mode = self.cores
        # ctl.configuration.solve.models = 0
        # ctl.configuration.solve.opt_mode = 'optN'
        ctl.load('induce.lp')
        for f in self.train:
            ctl.load(f)
        induction_program = [("base", []), ("class", [self.target_class]), ("classes_defs", []),
                             ("states", [self.states])]
           
        if self.exact:
            induction_program.append(("constraints_exact", []))
        else:
            induction_program.append(("constraints_heuristic", []))
        induction_program.append(("seq_end", []))
        ctl.ground(induction_program)
        # asynchronous version in order to use time_limit
        with ctl.solve(on_model=self.on_model, async_=True) as handle:
            handle.wait(self.time)
            handle.cancel()

    def test_model(self):
        ctl = clingo.Control()
        ctl.load('infer.lp')
        ctl.load(self.test)
        ctl.add("base", [], self.current_model)
        inference_program = [("base", []), ("class", [self.target_class]), ("classes_defs", [])]
        inference_program.append(("seq_end", []))
        ctl.ground(inference_program)
        ctl.solve(on_model=self.show_test_results)



    """
    def on_model(self, model):
        for atom in model.symbols(atoms=True):
            print(atom)
            debug = "stop"
    """
 

if __name__ == "__main__":

    # different datasets
    
    # train = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Earthquakes/folds/fold_1/Earthquakes_TRAIN_SAX_20_None_ASP.csv'
    # test = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Earthquakes/folds/fold_1/Earthquakes_TEST_SAX_20_None_ASP.csv'

    #train = '/home/nkatz/dev/Time-Series-SAX/data_coffee_compressed/90%_compressed_folds/fold_5/Coffee_TRAIN_SAX_20_None_ASP.csv'
    #test = '/home/nkatz/dev/Time-Series-SAX/data_coffee_compressed/90%_compressed_folds/fold_5/Coffee_TEST_SAX_20_None_ASP.csv'

    # small bio
    # train = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_5/ALIVE_TRAIN_SAX_20_None_ASP.csv'
    # test = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_5/ALIVE_TEST_SAX_20_None_ASP.csv'

    # MTS dataset

    # -----  nkatz current dataset path    --------
    #train_1 = '/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_1/MTS_TRAIN_SAX_8_ASP.csv'
    #train = [train_1]
    #test = '/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_1/MTS_TEST_SAX_8_ASP.csv'

    """ 
    train_1 = '/home/nkatz/fsm-learn-datasets/Haptics/folds/5/fold_0/Haptics_TRAIN_SAX_8_ASP.csv'
    train = [train_1]
    test = '/home/nkatz/fsm-learn-datasets/Haptics/folds/5/fold_0/Haptics_TEST_SAX_8_ASP.csv'
    """

    # large bio
    """
    train = '/home/nkatz/dev/Time-Series-SAX-LARGE-BIO-DATASET/ts-datasets/data/BioArchive/BioLarge/folds/fold_1' \
            '/ALIVE_TRAIN_SAX_20_None_ASP.csv'
    test = '/home/nkatz/dev/Time-Series-SAX-LARGE-BIO-DATASET/ts-datasets/data/BioArchive/BioLarge/folds/fold_1' \
           '/ALIVE_TEST_SAX_20_None_ASP.csv'
    """

    # large bio compressed
    # train = '/home/nkatz/dev/Time-Series-SAX/bio_compressed_folds/fold_1/ALIVE_TRAIN_SAX_8_None_ASP.csv'
    # test = '/home/nkatz/dev/Time-Series-SAX/bio_compressed_folds/fold_1/ALIVE_TEST_SAX_8_None_ASP.csv'
    


    # ----- mine/Christos current dataset path -----
    train_1 = 'C:/Users/xrist/myProjects/intern/Dataset/MTS_TRAIN_SAX_8_ASP.csv'
    train = [train_1]
    test = 'C:/Users/xrist/myProjects/intern/Dataset/MTS_TEST_SAX_8_ASP.csv'
    
    
    # ------------------------------------ RUNS: examples ------------------------------------
    # keeping a subset of cells type 
    """
    include_list=['apoptotic']
    mts_subsets(train_1,test, include_list)
    train_1='train.csv'
    train=[train_1]
    test='test.csv'
    """
    
    # ------------ 1. Runs: all parameters ------------
    """
    target_class, exact, incremental_mode, t_slice, incremental_search, chunk_size, test_threshold, analytics =  1, False, False, 0, False, 0, 0, True
    all_states, timeLimits = [2, 3, 4, 5], [60*5,60*10,60*15, 60*15]
    call_app_all_parameters(train_1, test, all_states, timeLimits, target_class, exact, incremental_mode, t_slice,  incremental_search, chunk_size , test_threshold, analytics)

    # ------------ 2. Runs: Incremental learning ------------
    chunk_size = 120
    test_threshold = 4
    file1 = open("incremental_learning_("+ str(chunk_size)+", "+ str(test_threshold)+").txt","w")
    file1.write("\n------------ Incremental learning: ("+ str(chunk_size)+", "+ str(test_threshold)+") ------------- \n")
    data = []
    csvReader = csv.reader(open(train_1, newline='\n'))
    for row in csvReader:
        row = ",".join(row)
        data.append(row)
    whole_len=len(data)
    #chunks_sizes, test_thresholds = [50, 50, 80, 80, 80, 100, 100, 100, 120, 120], [1,2, 2, 3, 4, 2 , 3, 4 , 3, 4] 
    #chunks_sizes = [50]
    
    time_list, train_vectors, train_scores, train_f1, test_vectors,  test_scores, test_f1 = [], [], [], [], [], [], []
    target_class, exact,incremental_mode, states, timeLimit, t_slice = 1, False, False, 2 ,1000, 0 

    file1.write("\n------------ incremental learning, best model (chunk size, test threshold = " + str(chunk_size) + ", "+ str(test_threshold) + " ): ------------- \n") 
    model, allBatches_model = call_solver_incremental_search(chunk_size, test_threshold, train_1, test, states,  target_class, exact, incremental_mode, t_slice, timeLimit)
    file1.write(str(model['best_model']) + "\n")
    file1.write("\n------------ incremental all learning, best model (chunk size, test threshold = " + str(chunk_size) + ", "+str(test_threshold) + " ) : ------------- \n") 
    file1.write(str(allBatches_model['best_model']) + "\n")
    train_scores.append(model['train_score'])
    train_scores.append(allBatches_model['train_score'])
    train_vectors.append(model['train_vector'])
    train_vectors.append(allBatches_model['train_vector'])
    train_f1.append(model['train_f1_accuracy'])
    train_f1.append(allBatches_model['train_f1_accuracy'])
    test_vectors.append(model['test_vector'])
    test_vectors.append(allBatches_model['test_vector'])
    test_scores.append(model['test_score'])
    test_scores.append(allBatches_model['test_score'])
    test_f1.append(model['test_f1_accuracy'])
    test_f1.append(allBatches_model['test_f1_accuracy'])
    time_list.append(model['time'])
    time_list.append(allBatches_model['time'])
    file1.write("\n------------ Added batches ------------- \n")
    file1.write("[ ")
    for batch in model['added_batches']:
        file1.write(str(batch) + ", ")
    file1.write("] \n")
    # percentage of dataset used
    per_batch = round(((model['batch_len']/whole_len)*100),2)
    file1.write("\n------------ Percentage of dataset used for batch learning: " + str(per_batch) + " % ------------- \n")
    file1.write("\n------------ Incremental_all learning revisions ------------- \n")
    file1.write("[ ")
    for batch in allBatches_model['train_batch']:
        file1.write(str(batch) + ", ")
    file1.write("] \n")


    # recording stats about runs: incremental_learning_(chunk_size,test_threshold).txt
    file1.write("\n------------ Train Vector ------------- \n")
    file1.write("[ ")
    for vector in train_vectors:
        file1.write(str(vector) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Train Score ------------- \n")
    file1.write("[ ")
    for score in train_scores:
        file1.write(str(score) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Train f1 ------------- \n")
    file1.write("[ ")
    for f1 in train_f1:
        file1.write(str(f1) + ", ")
    file1.write("] \n")
    

    file1.write("\n------------ Test Vector ------------- \n")
    file1.write("[ ")
    for vector in test_vectors:
        file1.write(str(vector) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Test Score ------------- \n")
    file1.write("[ ")
    for score in test_scores:
        file1.write(str(score) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Test f1 ------------- \n")
    file1.write("[ ")
    for f1 in test_f1:
        file1.write(str(f1) + ", ")
    file1.write("] \n")
    
    file1.write("\n------------ Time elapsed ------------- \n")
    file1.write("[ ")
    for cnt_time in time_list:
        file1.write(str(cnt_time) + ", ")
    file1.write("] \n")
    
    # ------------ 3. Runs: incremental_mode == True with whole dataset -----------------


    target_class, exact,incremental_mode, states, timeLimit  = 1, False, True, 2 ,1000 
    file1 = open("time_slices.txt","w")
    file1.write("\n------------ Time slices vs whole dataset ------------- \n")
    #time_slices=[11, 21, 31, 36, 41, 49]
    time_slices=[11, 21]
    all_time = 49
    models, train_vectors, train_scores, train_f1, train_earliness, test_vectors,  test_scores, test_f1,test_earliness = [], [], [], [], [], [], [], [], []
    
    file1 = open("time_slices.txt","a+")
    for t_slice in time_slices:
        print(t_slice)
        file1.write("\n------------ t = " + str(t_slice) + ", best model: ------------- \n") 
        data = call_solver(train_1, test, states,  target_class, exact = False, incremental_mode=incremental_mode, t_slice=t_slice, timeLimit=timeLimit)
        file1.write(str(data['best_model']) + "\n")
        models.append(data['best_model'])
        train_vectors.append(data['train_vector'])
        train_scores.append(data['train_score'])
        train_f1.append(data['train_f1_accuracy'])
        train_earliness.append(data['train_earliness'])
        test_vectors.append(data['test_vector'])
        test_scores.append(data['test_score'])
        test_f1.append(data['test_f1_accuracy'])
        test_earliness.append(data['test_earliness'])

    # recording stats about runs: time_slices.txt
    file1.write("\n------------ Train Vector ------------- \n")
    file1.write("[ ")
    for vector in train_vectors:
        file1.write(str(vector) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Train Score ------------- \n")
    file1.write("[ ")
    for score in train_scores:
        file1.write(str(score) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Train f1 ------------- \n")
    file1.write("[ ")
    for f1 in train_f1:
        file1.write(str(f1) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Train earliness ------------- \n")
    file1.write("[ ")
    for earliness in train_earliness:
        file1.write(str(earliness) + ", ")
    file1.write("] \n")

    file1.write("\n------------ Test Vector ------------- \n")
    file1.write("[ ")
    for vector in test_vectors:
        file1.write(str(vector) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Test Score ------------- \n")
    file1.write("[ ")
    for score in test_scores:
        file1.write(str(score) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Test f1 ------------- \n")
    file1.write("[ ")
    for f1 in test_f1:
        file1.write(str(f1) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Test earliness ------------- \n")
    file1.write("[ ")
    for earliness in test_earliness:
        file1.write(str(earliness) + ", ")
    file1.write("] \n")

    time_slices = [11,21,31,36,41,49]
    #learning_with_sliced_data(time_slices,train_1,test,states=2, exact=False, target_class=1, timeLimit=1000)

    # --------- 4. Runs: incremental learning && incremental mode ----------------
    
    time_slices=[31, 36, 41, 49]
    data = []
    csvReader = csv.reader(open(train_1, newline='\n'))
    for row in csvReader:
        row = ",".join(row)
        data.append(row)
    whole_len=len(data)
    all_time=49
    #chunks_sizes, test_thresholds = [50, 50, 80, 80, 80, 100, 100, 100, 120, 120, 150, 150 ], [1,2, 2, 3, 4, 2 , 3, 4 , 3, 4, 4,5 ] 
    #chunks_sizes = [50]
    chunk_size = 80
    test_thresholds = [3,3, 3, 3] 
    time_list, train_vectors, train_scores, train_f1, train_earliness, test_vectors,  test_scores, test_f1,test_earliness = [], [], [], [], [], [], [], [], []
    target_class, exact,incremental_mode, states, timeLimit = 1, False, True, 2 ,1000 
    file1 = open("incremental_learning(chunk size, test threshold)= (" + str(chunk_size) + ", "+ str(test_thresholds[0]) + ").txt","w")
    
    for (test_threshold, t_slice) in zip(test_thresholds,time_slices):
        print(chunk_size, test_threshold, t_slice)
        file1.write("\n------------ incremental learning, best model (chunk size, test threshold, t_slice) = ( " + str(chunk_size) + ", "+ str(test_threshold) +  ", " + str(t_slice) + " ): ------------- \n") 
        model, allBatches_model = call_solver_incremental_search(chunk_size, test_threshold, train_1, test, states,  target_class, exact, incremental_mode, t_slice, timeLimit)
        file1.write(str(model['best_model']) + "\n")
        file1.write("\n------------ incremental all learning, best model (chunk size, test threshold, t_slice= " + str(chunk_size) + ", "+str(test_threshold) + ", " + str(t_slice) + " ) : ------------- \n") 
        file1.write(str(allBatches_model['best_model']) + "\n")
        train_scores.append(model['train_score'])
        train_scores.append(allBatches_model['train_score'])
        train_vectors.append(model['train_vector'])
        train_vectors.append(allBatches_model['train_vector'])
        train_f1.append(model['train_f1_accuracy'])
        train_f1.append(allBatches_model['train_f1_accuracy'])
        train_earliness.append(model['train_earliness'])
        train_earliness.append(allBatches_model['train_earliness'])
        test_vectors.append(model['test_vector'])
        test_vectors.append(allBatches_model['test_vector'])
        test_scores.append(model['test_score'])
        test_scores.append(allBatches_model['test_score'])
        test_f1.append(model['test_f1_accuracy'])
        test_f1.append(allBatches_model['test_f1_accuracy'])
        test_earliness.append(model['test_earliness'])
        test_earliness.append(allBatches_model['test_earliness'])
        time_list.append(model['time'])
        time_list.append(allBatches_model['time'])
        file1.write("\n------------ Added batches ------------- \n")
        file1.write("[ ")
        for batch in model['added_batches']:
            file1.write(str(batch) + ", ")
        file1.write("] \n")
        # percentage of dataset used
        vertical=1
        if incremental_mode:
            vertical= t_slice/all_time
        per_batch, per_all = round(((model['batch_len']/whole_len)*vertical*100),2), round(vertical*100,2)
        file1.write("\n------------ Percentage of dataset used for batch learning: " + str(per_batch) + " and for incremental_all/all dataset learning " + str(per_all)+ " ------------- \n")
        file1.write("\n------------ Incremental_all learning revisions ------------- \n")
        file1.write("[ ")
        for batch in allBatches_model['train_batch']:
            file1.write(str(batch) + ", ")
        file1.write("] \n")
        
    # recording stats about runs: incremental_learning.txt
    

    file1.write("\n------------ Train Vector ------------- \n")
    file1.write("[ ")
    for vector in train_vectors:
        file1.write(str(vector) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Train Score ------------- \n")
    file1.write("[ ")
    for score in train_scores:
        file1.write(str(score) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Train f1 ------------- \n")
    file1.write("[ ")
    for f1 in train_f1:
        file1.write(str(f1) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Train earliness ------------- \n")
    file1.write("[ ")
    for earliness in train_earliness:
        file1.write(str(earliness) + ", ")
    file1.write("] \n")

    file1.write("\n------------ Test Vector ------------- \n")
    file1.write("[ ")
    for vector in test_vectors:
        file1.write(str(vector) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Test Score ------------- \n")
    file1.write("[ ")
    for score in test_scores:
        file1.write(str(score) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Test f1 ------------- \n")
    file1.write("[ ")
    for f1 in test_f1:
        file1.write(str(f1) + ", ")
    file1.write("] \n")
    file1.write("\n------------ Test earliness ------------- \n")
    file1.write("[ ")
    for earliness in test_earliness:
        file1.write(str(earliness) + ", ")
    file1.write("] \n")

    file1.write("\n------------ Time elapsed ------------- \n")
    file1.write("[ ")
    for cnt_time in time_list:
        file1.write(str(cnt_time) + ", ")
    file1.write("] \n")
    
# --------- 5. Runs: allStates && incremental learning && incremental mode ----------------
csvReader = csv.reader(open(train_1, newline='\n'))
data=[]
for row in csvReader:
    row = ",".join(row)
    data.append(row)
whole_len=len(data)
print("whole_len ", whole_len)
#target_class, exact, incremental_mode, incremental_search, chunk_size, test_threshold, analytics =  1, False, True,  True, 50, 3, False
target_class, exact, incremental_mode, incremental_search, chunk_size, test_threshold, analytics =  1, False, True,  False, 0, 0, False
#all_states, timeLimits, time_slices = [2, 3, 4, 5], [60*5,60*5,60*5, 60*5], [11,21, 31, 36,41,49]


all_states, timeLimits, time_slices = [5], [[2*60],[3*60],[4*60],[5*60],[6*60],[8*60]], [11,21, 31, 36,41,49]
file1 = open("all_learning_and_t_slices_necrotic.txt","w")

#all_states, timeLimits, time_slices = [2, 3, 4, 5], [60*5,60*5,60*5, 60*5], [31]
#all_states, timeLimits, time_slices = [2, 3, 4, 5], [20,35,45, 50], [31]
#all_states, timeLimits, time_slices = [2], [20], [10]
all_time=49
cnt=0
for t_slice in time_slices:
    vertical=1
    if incremental_mode:
        vertical= t_slice/all_time
    per_batch = round((vertical*100),2)
    file1.write(" \n----- time slices = " + str(t_slice) + " -----\n")
    file1.write("\n ----- Best incremental learning model, using " + str(per_batch) + " % of the dataset -----\n")
    print(t_slice)
    batching, best_states = call_app_all_parameters(train_1, test, all_states, timeLimits[cnt], target_class, exact, incremental_mode, t_slice,  incremental_search, chunk_size , test_threshold, analytics)
    
    
    file1.write(batching['best_model']+ "\n")
    file1.write("\nTest vectors: "+ str(batching['test_vector'])+ ", "+ str(batching['test_score']) +", " + str(round((1-(batching['test_score']/387))*100,2)) + ";  Train vectors: " + str(batching['train_vector']) + ", "+ str(batching['train_score']) + ", " +str(round((1-(batching['train_score']/1545))*100,2)) +  "\nTime elapsed: " + str(round(batching['time'],2)) +"\n" )
    cnt=cnt+1    

# --------- 6. Runs: Blind Stopping time  ---------------- 
file1 = open("blind_stopping_time_apoptotic2.txt","w")
target_class, exact, incremental_mode, incremental_search, chunk_size, test_threshold, analytics,t_slice =  1, False, False,  False, 0, 0, False, 0
states, timeLimit=5,480
print("start searching")
start=time.time()
app = App(train, test, states,  target_class, exact, incremental_mode, t_slice, time=timeLimit)
app.induce()
print(" Time elapsed: ", time.time()-start)
best_model, cur_size = app.best_model, app.model_size
time_slices = [11,21, 31, 36,41,49]
file1.write(" Best Model, using " + str(states) + " states:\n")
file1.write(str(best_model)+ "\n")
for t_slice in time_slices:
    print(t_slice)
    file1.write("--------- time slice = " + str(t_slice) + " ---------\n")
    data = []
    csvReader = csv.reader(open(test, newline='\n'))
    for row in csvReader:
        row = ",".join(row)
        data.append(row)
    cur_dataset=get_dataset_sliced_until_time_t(data,t_slice)
    with open("sliced_test_set.csv", 'w') as myfile:
        wr = csv.writer(myfile,delimiter="\n")
        wr.writerow(cur_dataset)
    app = App(train, "sliced_test_set.csv", states, target_class, exact, incremental_mode, time=timeLimit, only_test=True, saved_model=best_model, saved_model_size=cur_size)
    app.induce()
    print(app.test_vector)
    file1.write(" Score on test set:  " + str(app.best_test_score) + ", "+ str(round((1-(app.best_test_score/129)),4))+ "; Test earliness:" +str(app.test_earliness) + " \n")
    file1.write(" f1 - accuracy on test set:  " + str(app.f1_accuracy)+"\n")

"""

