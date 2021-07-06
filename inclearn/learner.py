import clingo
from asp_code import *
from utils import *
import sys
import multiprocessing

cores = multiprocessing.cpu_count()


class Tester:
    def __init__(self):
        pass

    """
    def f1(self, model):
        tps, fps, fns = 0, 0, 0
        for atom in model.symbols(atoms=True):
    """

    def infer(self, example, learnt_model):
        asp_program = learnt_model + '\n\n' + example + '\n\n' + inference_program
        ctl = clingo.Control()
        ctl.configuration.solve.parallel_mode = cores
        ctl.add("base", [], asp_program)
        ctl.ground([("base", [])], context=self)
        ctl.solve(on_model=self.f1)


class Learner:

    def __init__(self):
        pass

    current_model = ''

    @staticmethod
    def show_model(model):
        print(model)

    def on_model(self, model):
        print("Found model:\n{0}".format(model))
        self.current_model = ' '.join(map(lambda m: m + '.', format(model).split(' ')))

    def run_task(self, task, example, learnt_model=""):
        if task == "train":
            asp_program = self.current_model + '\n\n' + example + '\n\n' + learning_program
            on_model_function = self.on_model
        else:
            asp_program = learnt_model + '\n\n' + example + '\n\n' + inference_program
            on_model_function = self.show_model

        ctl = clingo.Control()
        ctl.configuration.solve.parallel_mode = cores
        ctl.add("base", [], asp_program)
        ctl.ground([("base", [])], context=self)

        if ctl.solve().unsatisfiable:
            print('UNSATISFIABLE')
            sys.exit(0)
        ctl.solve(on_model=on_model_function)


def inner_loop(pos_string, negative_strings):
    """
    Given a positive string s+, learns an initial automaton A from the pair (s+, s1-),
    where s1- is the first negative string, and then refines A w.r.t. to all remaining
    negative strings.
    :return: A learnt automaton
    """
    print("Running inner loop...")
    _learner = Learner()

    pairs = [pos_string + negString for negString in negative_strings]
    counter = 1
    for p in pairs:
        print(counter)
        _learner.run_task("train", p)
        counter = counter + 1
    print("Finished inner loop. Learnt automaton:\n" + _learner.current_model)
    return _learner.current_model


# def set_cover_loop():


if __name__ == "__main__":
    training_data_path = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/HandOutlines' \
                         '/HandOutlines_TRAIN_SAX_20_ASP.csv'
    # training_data_path = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Coffee' \
    #                     '/Coffee_TRAIN_SAX_20_ASP.csv'
    positive, negative = split_data(training_data_path)
    print(len(positive), len(negative))

    # Learn an initial model that covers the first positive and no negatives
    seed = positive[0]
    model = inner_loop(positive[0], negative)

    # Test that no negatives are covered
    print("Testing...")
    learner = Learner()
    # for n in negative:
    #    learner.run_task("infer", n, model)

    # for n in positive:
    #    learner.run_task("infer", n, model)

    learner.run_task("infer", seed, model)

    # done = False
    # while not done:
    # Find the positives that are covered by the current model:
