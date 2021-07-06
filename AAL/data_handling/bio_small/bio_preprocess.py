train_data_whole = '/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_4/MTS_TRAIN_SAX_8_ASP.csv'
train_alive_pos = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_4/pos_alive', 'w')
train_alive_neg = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_4/neg_alive', 'w')
train_necrotic_pos = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_4/pos_necrotic', 'w')
train_necrotic_neg = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_4/neg_necrotic', 'w')
train_apoptotic_pos = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_4/pos_apoptotic', 'w')
train_apoptotic_neg = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_4/neg_apoptotic', 'w')
# train_heading_pos = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_0/pos_heading', 'w')
# train_heading_neg = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_0/neg_heading', 'w')
# train_course_pos = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_0/pos_course', 'w')
# train_course_neg = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_0/neg_course', 'w')

file1 = open(train_data_whole, 'r')
lines = file1.readlines()

for line in lines:
    label = line[len(line) - 4]
    if "alive" in line:
        if label == '1':
            train_alive_pos.write(line)
        else:
            train_alive_neg.write(line)
    elif "necrotic" in line:
        if label == '1':
            train_necrotic_pos.write(line)
        else:
            train_necrotic_neg.write(line)
    elif "apoptotic" in line:
        if label == '1':
            train_apoptotic_pos.write(line)
        else:
            train_apoptotic_neg.write(line)
    else:
        pass

train_alive_pos.close()
train_alive_neg.close()
train_necrotic_pos.close()
train_necrotic_neg.close()
train_apoptotic_pos.close()
train_apoptotic_neg.close()
