import os
import os.path
import shutil

# LABELS = {
#     '0': '0', # person        
#     '1': '1', # bicycle       
#     '2': '2', # car
#     '3': '3', # motorcycle    
#     '5': '4', # bus
#     '7': '5', # truck
#     '11': '6', # stop sign    
#     '15': '7', # cat
#     '16': '8', # dog
#     '17': '9', # horse
#     '18': '10', # sheep
#     '19': '11', # cow
#     '29': '12', # frisbee
#     '32': '13', # sports ball
#     '36': '14', # skateboard
# }

LABELS = {
    '0': '15', # GTL
    '1': '16', # RTL
    '2': '17', # YTL
    '3': '1', # bike
    '4': '4', # bus
    '5': '2', # car
    '6': '18', # construction 
    '7': '8', # dog 
    '8': '3', # motorcyle
    '9': '0', # person
    '10': '19', # scooter
    '11': '6', # stop
    '12': '21', # traffic barrel
    '13': '22', # trash
    '14': '5', # truck
}

def filter_files(img_path: str,
                 label_path: str,
                 new_img_path: str,
                 new_label_path: str,
                 txt_file: str,
                 txt_file_path: str) -> None:
    txt_file_data = ''
    i, size = 0, len(os.listdir(img_path))
    for img_file in os.listdir(img_path):
        print(f'{i} / {size}')
        i += 1
        if not os.path.isfile(img_path + img_file):
            continue

        new_label_data = ''

        label_file = img_file[:-3] + 'txt'
        if not os.path.isfile(label_path + label_file):
            continue

        with open(label_path + label_file, 'r') as f:
            for label_data in f.readlines():
                seg_data = label_data.split(' ')
                data_class = seg_data[0]
                if not data_class in LABELS:
                    continue

                seg_data[0] = LABELS[data_class]
                new_label_data += ' '.join(seg_data)
        if not new_label_data:
            continue

        txt_file_data += txt_file_path + img_file + '\n'

        shutil.copyfile(img_path + img_file,
                        new_img_path + img_file)
        
        with open(new_label_path + label_file, 'w') as f:
            f.write(new_label_data)
    
    with open(txt_file, 'w') as f:
        f.write(txt_file_data)

# Train.
print('Train')
filter_files(
    './datasets/robo_gator/train/images/',
    './datasets/robo_gator/train/labels/',
    './datasets/gator/images/train/',
    './datasets/gator/labels/train/',
    './datasets/gator/train.txt',
    './images/train/'
)

print()

# Validation.
print('Validation')
filter_files(
    './datasets/robo_gator/valid/images/',
    './datasets/robo_gator/valid/labels/',
    './datasets/gator/images/val/',
    './datasets/gator/labels/val/',
    './datasets/gator/val.txt',
    './images/val/'
)
