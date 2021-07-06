train_data_whole = '/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/Maritime_TRAIN_SAX_8_ASP.csv'
train_lat_pos = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/pos_lat', 'w')
train_lat_neg = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/neg_lat', 'w')
train_lon_pos = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/pos_lon', 'w')
train_lon_neg = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/neg_lon', 'w')
train_speed_pos = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/pos_speed', 'w')
train_speed_neg = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/neg_speed', 'w')
train_heading_pos = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/pos_heading', 'w')
train_heading_neg = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/neg_heading', 'w')
train_course_pos = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/pos_course', 'w')
train_course_neg = open('/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/neg_course', 'w')

# TODO clear files when run this.

file1 = open(train_data_whole, 'r')
lines = file1.readlines()

for line in lines:
    label = line[len(line) - 4]
    if "longitude" in line:
        if label == '1':
            train_lon_pos.write(line)
        else:
            train_lon_neg.write(line)
    elif "latitude" in line:
        if label == '1':
            train_lat_pos.write(line)
        else:
            train_lat_neg.write(line)
    elif "speed" in line:
        if label == '1':
            train_speed_pos.write(line)
        else:
            train_speed_neg.write(line)
    elif "heading" in line:
        if label == '1':
            train_heading_pos.write(line)
        else:
            train_heading_neg.write(line)
    elif "course_over_ground" in line:
        if label == '1':
            train_course_pos.write(line)
        else:
            train_course_neg.write(line)
    else:
        pass

train_lon_pos.close()
train_lon_neg.close()
train_lat_pos.close()
train_lat_neg.close()
train_speed_pos.close()
train_speed_neg.close()
train_heading_pos.close()
train_heading_neg.close()
train_course_pos.close()
train_course_neg.close()
