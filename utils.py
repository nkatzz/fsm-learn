def grouped(list, n):
    """Splits list into sublists of length n"""
    return [list[i:i + n] for i in range(0, len(list), n)]


def split_data(training_path):
    """
    Splits the training data into positive and negative subsets.
    :param training_path: The path to the training data.
    :return: A 2-tuple of lists (positive, negative)
    """

    def is_positive(string):
        """Helper function."""
        classatom = string.split(" ").pop()
        classvalue = classatom.split(",")[1].split(")")[0]
        return True if classvalue == '1' else False

    positive, negative = [], []
    with open(training_path) as training_data:
        lines = training_data.readlines()
        for x in lines:
            positive.append(x) if is_positive(x) else negative.append(x)
    return positive, negative


def atoms_to_string_seqs_rpni(training_path):
    """
    Converts the data to a proper format to use by RPNI
    :return:
    """

    def preds_to_string(predicate_sequence):
        predicates = predicate_sequence.split(". ")
        predicates.pop()  # discard the last element, it is the class.
        string_seq = ''.join(map(lambda x: x.split(',')[1], predicates))
        return string_seq

    pos, neg = split_data(training_path)
    positive_strings = [preds_to_string(x) for x in pos]
    negative_strings = [preds_to_string(x) for x in neg]
    return positive_strings, negative_strings


if __name__ == "__main__":
    """
    t = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Coffee/Coffee_TRAIN_SAX_20_ASP.csv'
    pos, neg = atoms_to_string_seqs_rpni(t)
    for p in pos:
        print(p)
    print('negs')
    for p in neg:
        print(p)
    """

    """
    training_data_path = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/HandOutlines' \
                         '/HandOutlines_TRAIN_SAX_20_ASP.csv'
    pos_file = "/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/HandOutlines/RPNI-positive-strings"
    neg_file = "/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/HandOutlines/RPNI-negative-strings"

    training_data_path = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Coffee' \
                         '/Coffee_TRAIN_SAX_20_ASP.csv'
    pos_file = "/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Coffee/RPNI-positive-strings"
    neg_file = "/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/UCRArchive_2018/Coffee/RPNI-negative-strings"
    """

    """
    training_data_path = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_1' \
                         '/ALIVE_TRAIN_SAX_20_ASP.csv'
    pos_file = "/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_5/RPNI-positive-strings"
    neg_file = "/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_5/RPNI-negative-strings"
    """

    """
    training_data_path = '/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_1' \
                         '/NECROTIC_TRAIN_SAX_20_None_ASP.csv'
    pos_file = "/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_2/RPNI/necrotic/train-positives"
    neg_file = "/home/nkatz/dev/Time-Series-SAX/ts-datasets/data/BioArchive/folds/fold_2/RPNI/necrotic/train-negatives"
    """

    training_data_path = '/home/nkatz/dev/Time-Series-SAX-LARGE-BIO-DATASET/ts-datasets/data/BioArchive/BioLarge/folds/fold_1' \
                         '/ALIVE_TRAIN_SAX_20_None_ASP.csv'
    pos_file = "/home/nkatz/dev/Time-Series-SAX-LARGE-BIO-DATASET/ts-datasets/data/BioArchive/BioLarge/folds/fold_1/RPNI/alive/train-positives"
    neg_file = "/home/nkatz/dev/Time-Series-SAX-LARGE-BIO-DATASET/ts-datasets/data/BioArchive/BioLarge/folds/fold_1/RPNI/alive/train-negatives"

    pos, neg = atoms_to_string_seqs_rpni(training_data_path)

    f_pos = open(pos_file, "a")
    f_neg = open(neg_file, "a")
    for x in pos:
        f_pos.write(x + "\n")
    f_pos.close()

    for x in neg:
        f_neg.write(x + "\n")
    f_neg.close()
