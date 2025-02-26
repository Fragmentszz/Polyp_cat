import os

def get_results_from_name(name):
    # 0_0.95848149061203_0.9584815502166748_0.9202731251716614.png
    name = name.replace('.png','')
    tmp = name.split('_')
    return tmp[0],float(tmp[1]),float(tmp[2]),float(tmp[3])

def get_results_from_dir(dir):
    res = []
    for name in os.listdir(dir):
        res.append(get_results_from_name(name))
    return res
