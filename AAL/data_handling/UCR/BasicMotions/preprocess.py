train_data_whole = '/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/BasicMotions_TRAIN_SAX_8_ASP.csv'
train_feature1_0 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature1_0', 'w')
train_feature1_1 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature1_1', 'w')
train_feature1_2 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature1_2', 'w')
train_feature1_3 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature1_3', 'w')

train_feature2_0 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature2_0', 'w')
train_feature2_1 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature2_1', 'w')
train_feature2_2 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature2_2', 'w')
train_feature2_3 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature2_3', 'w')

train_feature3_0 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature3_0', 'w')
train_feature3_1 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature3_1', 'w')
train_feature3_2 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature3_2', 'w')
train_feature3_3 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature3_3', 'w')

train_feature4_0 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature4_0', 'w')
train_feature4_1 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature4_1', 'w')
train_feature4_2 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature4_2', 'w')
train_feature4_3 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature4_3', 'w')

train_feature5_0 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature5_0', 'w')
train_feature5_1 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature5_1', 'w')
train_feature5_2 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature5_2', 'w')
train_feature5_3 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature5_3', 'w')

train_feature6_0 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature6_0', 'w')
train_feature6_1 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature6_1', 'w')
train_feature6_2 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature6_2', 'w')
train_feature6_3 = open('/home/nkatz/dev/datasets_asp_wayeb_04062021/selected_UCR/BasicMotions/feature6_3', 'w')

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
        else:
            train_feature1_3.write(line)
    elif "feature2" in line:
        if label == '0':
            train_feature2_0.write(line)
        elif label == '1':
            train_feature2_1.write(line)
        elif label == '2':
            train_feature2_2.write(line)
        else:
            train_feature2_3.write(line)
    elif "feature3" in line:
        if label == '0':
            train_feature3_0.write(line)
        elif label == '1':
            train_feature3_1.write(line)
        elif label == '2':
            train_feature3_2.write(line)
        else:
            train_feature3_3.write(line)
    elif "feature4" in line:
        if label == '0':
            train_feature4_0.write(line)
        elif label == '1':
            train_feature4_1.write(line)
        elif label == '2':
            train_feature4_2.write(line)
        else:
            train_feature4_3.write(line)
    elif "feature5" in line:
        if label == '0':
            train_feature5_0.write(line)
        elif label == '1':
            train_feature5_1.write(line)
        elif label == '2':
            train_feature5_2.write(line)
        else:
            train_feature5_3.write(line)
    elif "feature6" in line:
        if label == '0':
            train_feature6_0.write(line)
        elif label == '1':
            train_feature6_1.write(line)
        elif label == '2':
            train_feature6_2.write(line)
        else:
            train_feature6_3.write(line)
    else:
        pass

train_feature1_0.close()
train_feature1_1.close()
train_feature1_2.close()
train_feature1_3.close()

train_feature2_0.close()
train_feature2_1.close()
train_feature2_2.close()
train_feature2_3.close()

train_feature3_0.close()
train_feature3_1.close()
train_feature3_2.close()
train_feature3_3.close()

train_feature4_0.close()
train_feature4_1.close()
train_feature4_2.close()
train_feature4_3.close()

train_feature5_0.close()
train_feature5_1.close()
train_feature5_2.close()
train_feature5_3.close()

train_feature6_0.close()
train_feature6_1.close()
train_feature6_2.close()
train_feature6_3.close()
