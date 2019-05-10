import os
Path = './sample_data/joblib/'
for filename in os.listdir('./sample_data/joblib/'):
    if filename == 'test.joblib':
        continue
    tmp1 = filename.split("_")[0]
    tmp2 = filename.split("_")[1]
    tmp1 += '1'
    re_filename = tmp1 + '_' + tmp2
    os.rename(Path+filename, Path+re_filename)
    print(Path+re_filename)