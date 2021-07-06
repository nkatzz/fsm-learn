import clingo
from asp_code import *
from utils import *


class App:
    current_model = ''
    ctl = clingo.Control()

    def __init__(self, task):
        self.init_control(task)

    def init_control(self, task):
        if task == "train":
            self.ctl.add("base", [], learning_program_ec)

    def on_model(self, model):
        print("Found model:\n{0}".format(model))
        self.current_model = ' '.join(map(lambda x: x + '.', format(model).split(' ')))

    """
    def on_model(self, model):
        for atom in model.symbols(atoms=True):
            print(atom)
            debug = "stop"
    """

    def learn_from_batch(self, example):
        # asp_input = self.current_model + '\n\n' + example + '\n\n' + learning_program
        # ctl = clingo.Control()

        asp_input = example
        self.ctl.add("base", [], asp_input)
        self.ctl.ground([("base", [])], context=self)
        self.ctl.solve(on_model=self.on_model)

    def show_model(self, model):
        print(model)

    def infer(self, example, model):
        """Test a model on an example"""
        asp_input = model + '\n\n' + example + '\n\n' + inference_program_ec
        ctl = clingo.Control()
        ctl.add("base", [], asp_input)
        ctl.ground([("base", [])], context=self)
        ctl.solve(on_model=self.show_model)


def train(training_path):
    app = App("train")
    with open(training_path) as training_data:
        lines = training_data.readlines()
        chunks = grouped(lines, 5)
        counter = 1
        for mini_batch in chunks:
            print('Batch {0}'.format(counter))
            mini_batch_data = ' '.join(mini_batch)
            app.learn_from_batch(mini_batch_data)
            counter = counter + 1
    print('\nFinished training. Final learnt automaton:\n' + app.current_model)


def test(testing_path, model):
    app = App("test")
    with open(testing_path) as testing_data:
        lines = testing_data.readlines()
        chunks = grouped(lines, 5)
        counter = 1
        for mini_batch in chunks:
            print('Batch {0}'.format(counter))
            mini_batch_data = ' '.join(mini_batch)
            app.infer(mini_batch_data, model)
            counter = counter + 1
    print('\nFinished testing.')


if __name__ == "__main__":
    training = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/HandOutlines' \
               '/HandOutlines_TRAIN_SAX_20_ASP.csv'
    testing = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/HandOutlines' \
              '/HandOutlines_TEST_SAX_20_ASP.csv'

    train(training)

    _model = 'transition(1,s,2). transition(1,t,2). transition(2,s,1). transition(1,f,2). transition(1,n,' \
             '2). transition(2,f,1). transition(2,n,1). transition(2,t,1). transition(1,j,2). transition(2,j,' \
             '1). transition(2,l,1). transition(1,b,2). transition(1,k,2). transition(2,b,1). transition(2,k,' \
             '1). transition(1,h,2). transition(2,h,1). transition(1,l,2). transition(1,p,2). transition(1,m,' \
             '2). transition(2,m,1). transition(1,c,2). transition(2,c,1). transition(2,g,1). transition(1,g,' \
             '2). transition(1,i,2). transition(1,q,2). transition(2,i,1). transition(2,p,1). transition(2,q,' \
             '1). transition(1,r,2). transition(1,a,2). transition(2,a,1). transition(2,r,1). transition(1,d,' \
             '2). transition(2,d,1). transition(1,e,2). transition(2,e,1). transition(2,o,1). final(1). '

    # _model = 'transition(1,p,2). transition(2,p,1). transition(1,c,2). transition(2,c,1). final(2). final(1).'

    test(training, _model)
