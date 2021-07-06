train_data_whole = '/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/DodgerLoopDay/DodgerLoopDay_TRAIN_SAX_20_ASP.csv'
train_feature1_0 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/DodgerLoopDay/feature1_0', 'w')
train_feature1_1 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/DodgerLoopDay/feature1_1', 'w')
train_feature1_2 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/DodgerLoopDay/feature1_2', 'w')
train_feature1_3 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/DodgerLoopDay/feature1_3', 'w')
train_feature1_4 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/DodgerLoopDay/feature1_4', 'w')
train_feature1_5 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/DodgerLoopDay/feature1_5', 'w')
train_feature1_6 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/DodgerLoopDay/feature1_6', 'w')

file1 = open(train_data_whole, 'r')
lines = file1.readlines()

for line in lines:
    label = line[len(line) - 4]
    if "feature1" in line:
        if label == '0':
            train_feature1_0.write(line)
        elif label == '1':
            train_feature1_1.write(line)
        elif label == '2':
            train_feature1_2.write(line)
        elif label == '3':
            train_feature1_3.write(line)
        elif label == '4':
            train_feature1_4.write(line)
        elif label == '5':
            train_feature1_5.write(line)
        elif label == '6':
            train_feature1_6.write(line)
        else:
            pass
    else:
        pass

train_feature1_0.close()
train_feature1_1.close()
train_feature1_2.close()
train_feature1_3.close()
train_feature1_4.close()
train_feature1_5.close()
train_feature1_6.close()
