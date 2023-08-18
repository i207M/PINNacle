import json
import math

def avg(l):
    return sum(l)/len(l)

def std(l):
    a = avg(l)
    return math.sqrt(sum([(x-a)**2 for x in l]) / len(l))

catagories = ["burger", "chaotic", "heat", "ns", "poisson", "wave", "inverse"]
metric_cats = ["time", "trainloss", "l2rel", "l1rel", "mse", "mae", "maxe", "csve", "f_low", "f_mid", "f_high"]
conf_file = json.load(open("run_all_config.json"))
buffer = list()
for cat in catagories:
    conf_cat = conf_file[cat]
    for casename,conf in conf_cat.items():
        metric_lists =[list() for _ in metric_cats]
        try:
            for i in range(0,3):
                f_fb = open("../benchmark_results/"+"fb"+"/"+casename+"_"+str(i))
                fb_content = f_fb.readline(); f_fb.close()
                fb_content = fb_content.split()
                for j in range(len(metric_cats)):
                    metric_lists[j].append(float(fb_content[j]))
            metric_stats = [(avg(metric_lists[j]),std(metric_lists[j])) for j in range(len(metric_cats))]
            str_buf = [casename]
            for j in range(len(metric_cats)):
                m = metric_stats[j]
                str_buf += ["%.6g"%m[0], "%.6g"%m[1]]
            str_buf += ["*".join(fb_content[11].split("_")), "*".join(str(_) for _ in conf["ba"])]
            buffer.append(str_buf)
            
        except FileNotFoundError:
            print("results for case "+casename+" not found")

f_csv = open("allmetric.csv", 'w')
frow = ["pde"]
for j in range(len(metric_cats)):
    frow += [metric_cats[j]+"_avg", metric_cats[j]+"_std"]
frow += ["grid", "batch_size"]
buffer.insert(0, frow)
for entry in buffer:
    f_csv.write(", ".join(entry)+'\n')
f_csv.close()