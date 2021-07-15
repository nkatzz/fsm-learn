import clingo
import multiprocessing
import time


def show_model(model,best=True):
    if(best):
        print("Best Model: ")
    print(model)


def best_N_models(models, N=1,metric='accuracy_metric'):
    i=1
    if N >1:
        print("Best {0} models: ".format(N))
    for model in sorted(models, key = lambda i: (i[metric]))[:N]:
        if N >1:
            print("\nModel {0}\n{1}".format(i, model['current_model']))
        else:
            print("\nBest Model \n {0}".format(model['current_model']))

        print("Training accuracy: {0}. More specifically (FN,FP,size): {1} " .format(model[metric],model['cost']))
        print('Performance (testing set):\nTPs: {0}, FPs: {1}, FNs: {2}, F1: {3}\nTP earlines: {4}, FP earliness: {5}\nEarliness: {6}, '
              'F1/earliness harmonic mean: {7}'.format(model['tps'], model['fps'], model['fns'], model['f1'], model['tp_earliness'], model['fp_earliness'], model['earliness'],
                                                       model['harmonic_mean']))
        i+=1


class App:
    # storing solutions
    models=[]
    cur_model={}
    best_training_acc=float('inf')

    current_model = ''
    best_model = ''
    best_test_f1 = 0.0
    model_couner = 1
    cores = multiprocessing.cpu_count()
    ctl = clingo.Control()

    def __init__(self, training_path, testing_path, states, exact, target_class, incremental=False, time=float('inf')):
        self.train = training_path
        self.test = testing_path
        self.exact = exact
        self.target_class = target_class
        self.states = states
        self.incremental_mode = incremental
        self.time=time
    
   

    def on_model(self, model):
        """ does nothing :(
        if(time.time() - start_time > self.time):
            print("stucked here")
            return
        """
        self.cur_model={}
        self.current_model = ' '.join(map(lambda x: x + '.', format(model).split(' ')))
        self.cur_model['current_model']=self.current_model
        #print("\nModel {0}\n{1}\nCost (training set): {2}".format(self.model_couner, self.current_model, model.cost))
        self.cur_model['cost']=model.cost
        print("\nModel {0} ".format(self.model_couner))
        # ----- issue: empty set enters here -----
        # training accuracy
        accuracy_metric=model.cost[0]+model.cost[1]+model.cost[2]
        self.cur_model['accuracy_metric']=accuracy_metric
        if  accuracy_metric < self.best_training_acc:
            self.best_training_acc = accuracy_metric
            self.best_model = self.current_model
        self.model_couner += 1
        if self.current_model != '.':
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
        # print(tp_earliness)
        # print(fp_earliness)
        appended = tp_earliness_vals + fp_earliness_vals
        # print(appended)
        earliness = float(sum(appended)) / len(appended) if len(appended) > 0 else 'None'
        harmonic_mean = 2 * f1 * earliness / (f1 + earliness) if earliness != 'None' else 'None'
        
        #print(self.cur_model['current_model']==self.current_model)
        # storing parameters of all models
        self.cur_model['tps'], self.cur_model['fps'], self.cur_model['fns'], self.cur_model['f1'], self.cur_model['tp_earliness'], self.cur_model['fp_earliness'], self.cur_model['earliness'], self.cur_model['harmonic_mean']=(
        tps, fps, fns, f1, tp_earliness, fp_earliness, earliness,harmonic_mean)
        self.models.append(self.cur_model)
        # printing models at get_N_best_models
        """  
        print('Performance (testing set):\nTPs: {0}, FPs: {1}, FNs: {2}, F1: {3}\nTP earlines: {4}, FP earliness: {5}\nEarliness: {6}, '
              'F1/earliness harmonic mean: {7}'.format(tps, fps, fns, f1, tp_earliness, fp_earliness, earliness,
        
                                               harmonic_mean))
    
        """
        """
        if f1 > self.best_test_f1:
            self.best_test_f1 = f1
            self.best_model = self.current_model
        print('TPs: {0}, FPs: {1}, FNs: {2}, F1: {3}\nBest F1-score: {4}\n Best model: {5}'.format(tps, fps, fns, f1,
                                                                                                   self.best_test_f1,
                                                                                                   self.best_model))
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

        if self.incremental_mode:
            induction_program.append(("seq_end_inc", [10]))
        else:
            induction_program.append(("seq_end", []))
        
        ctl.ground(induction_program)

        # asynchronous version in order to use time_limit
        with ctl.solve(on_model=self.on_model, async_=True) as handle:
            handle.wait(self.time)
            handle.cancel()
            print (handle.get())
        #ctl.solve(on_model=self.on_model)

    def test_model(self):
        ctl = clingo.Control()
        ctl.load('infer.lp')
        ctl.load(self.test)
        ctl.add("base", [], self.current_model)
        inference_program = [("base", []), ("class", [self.target_class]), ("classes_defs", [])]

        if self.incremental_mode:
            inference_program.append(("seq_end_inc", [10]))
        else:
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
   
    # ----- mine/Christos current dataset path -----
    train_1 = 'C:/Users/xrist/myProjects/intern/Dataset/MTS_TRAIN_SAX_8_ASP.csv'
    train = [train_1]
    test = 'C:/Users/xrist/myProjects/intern/Dataset/MTS_TEST_SAX_8_ASP.csv'

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
    
    # defining parameters

    target_class = 1
    exact = False
    incremental_mode = False  # True
    allStates=[2,3,4,5]
    best_overall_model=''
    timeLimits=[60,120,180,200]
    best_score=float('inf')
    models=[]
    for (states,timeLimit) in zip(allStates,timeLimits):
        app = App(train, test, states, exact, target_class, incremental_mode, time=timeLimit)
        start_time = time.time()
        app.induce()
        print("--- %s seconds ---" % (time.time() - start_time))
        # storing the model with minimized training accuracy
        if app.best_training_acc < best_score:
            best_model=app.best_model
            best_score=app.best_training_acc
        # storing all models
        for model in app.models:
            models.append(model)

    # printing only the model
    show_model(best_model)
    # printing all parameters concerned with the model
    best_N_models(models, N=3)
    