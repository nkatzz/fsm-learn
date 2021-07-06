import clingo
import multiprocessing
import time


def show_model(model):
    print(model)


class App:
    current_model = ''
    best_model = ''
    best_test_f1 = 0.0
    model_couner = 1
    cores = multiprocessing.cpu_count()
    ctl = clingo.Control()

    def __init__(self, training_path, testing_path, states, exact, target_class, incremental=False):
        self.train = training_path
        self.test = testing_path
        self.exact = exact
        self.target_class = target_class
        self.states = states
        self.incremental_mode = incremental

    def on_model(self, model):
        self.current_model = ' '.join(map(lambda x: x + '.', format(model).split(' ')))
        print("\nModel {0}\n{1}\ncost: {2}".format(self.model_couner, self.current_model, model.cost))
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
        print('TPs: {0}, FPs: {1}, FNs: {2}, F1: {3}\nTP earlines: {4}, FP earliness: {5}\nEarliness: {6}, '
              'F1/earliness harmonic mean: {7}'.format(tps, fps, fns, f1, tp_earliness, fp_earliness, earliness,
                                                       harmonic_mean))
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
        ctl.solve(on_model=self.on_model)

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
    # train = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Earthquakes/folds/fold_1/Earthquakes_TRAIN_SAX_20_None_ASP.csv'
    # test = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Earthquakes/folds/fold_1/Earthquakes_TEST_SAX_20_None_ASP.csv'

    #train = '/home/nkatz/dev/Time-Series-SAX/data_coffee_compressed/90%_compressed_folds/fold_5/Coffee_TRAIN_SAX_20_None_ASP.csv'
    #test = '/home/nkatz/dev/Time-Series-SAX/data_coffee_compressed/90%_compressed_folds/fold_5/Coffee_TEST_SAX_20_None_ASP.csv'

    # small bio
    # train = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_5/ALIVE_TRAIN_SAX_20_None_ASP.csv'
    # test = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_5/ALIVE_TEST_SAX_20_None_ASP.csv'

    """
    train_1 = '/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_0/MTS_TRAIN_SAX_8_ASP.csv'
    train = [train_1]
    test = '/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_0/MTS_TEST_SAX_8_ASP.csv'
    """

    train_1 = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Haptics/folds/fold_1/Haptics_TRAIN_SAX_10_500_ASP.csv'
    train = [train_1]
    test = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Haptics/folds/fold_1/Haptics_TEST_SAX_10_500_ASP.csv'

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

    target_class = 5
    exact = False
    states = 3
    incremental_mode = False  # True
    app = App(train, test, states, exact, target_class, incremental_mode)

    start_time = time.time()
    app.induce()
    print("--- %s seconds ---" % (time.time() - start_time))
