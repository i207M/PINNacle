import json
import math

def avg(l):
    return sum(l)/len(l)

def std(l):
    a = avg(l)
    return math.sqrt(sum([(x-a)**2 for x in l]) / len(l))

metric_cats = ["time", "trainloss", "l2rel", "l1rel", "mse", "mae", "maxe", "csve", "f_low", "f_mid", "f_high"]
parameter_vals = {"HeatMultiscaleExact": [5, 10, 20, 40],
                "WaveEquation1D": [2, 4, 6, 8, 10],
                "LidDrivenFlow":[2, 4, 8, 16, 32],
                "Poisson2D_Classic":[1, 2, 4, 8, 16]}

buffer = list()
for casename in ["HeatMultiscaleExact", "WaveEquation1D", "LidDrivenFlow", "Poisson2D_Classic"]:
    for hpval in parameter_vals[casename]:
        metric_lists =[list() for _ in metric_cats]
        try:
            for i in range(0,3):
                f_fb = open("../benchmark_results/parampde/"+casename+"_"+str(hpval)+"_"+str(i))
                fb_content = f_fb.readline(); f_fb.close()
                fb_content = fb_content.split()
                for j in range(len(metric_cats)):
                    metric_lists[j].append(float(fb_content[j]))
            metric_stats = [(avg(metric_lists[j]),std(metric_lists[j])) for j in range(len(metric_cats))]
            str_buf = [casename, str(hpval)]
            for j in range(len(metric_cats)):
                m = metric_stats[j]
                str_buf += ["%.6g"%m[0], "%.6g"%m[1]]
            buffer.append(str_buf)
            
        except FileNotFoundError:
            print("results for "+casename+" "+str(hpval)+" not found")

f_csv = open("parametric.csv", 'w')
frow = ["pde", "paramter"]
for j in range(len(metric_cats)):
    frow += [metric_cats[j]+"_avg", metric_cats[j]+"_std"]
buffer.insert(0, frow)
for entry in buffer:
    f_csv.write(", ".join(entry)+'\n')
f_csv.close()