import json
import math

def avg(l):
    return sum(l)/len(l)

def std(l):
    a = avg(l)
    return math.sqrt(sum([(x-a)**2 for x in l]) / len(l))

def tostr(v,read=True):
    if type(v) == list:
        return ("-" if read else "*").join(str(_) for _ in v)
    else:
        return str(v)

metric_cats = ["time", "trainloss", "l2rel", "l1rel", "mse", "mae", "maxe", "csve", "f_low", "f_mid", "f_high"]
hypername = "div"
hypervals_div = {"Burgers1D": [[1, 1], [2, 1], [3, 1], [1, 2]],
                "GrayScott": [[1, 1, 1], [1, 1, 3], [1, 1, 5], [2, 2, 1]],
                "HeatComplex":[[1, 1, 1], [1, 1, 3], [1, 1, 5], [2, 2, 1]],
                "Poisson2D_Classic":[[1, 1], [1, 2], [2, 1], [2, 2]]}

buffer = list()
for casename in ["Burgers1D", "GrayScott", "HeatComplex", "Poisson2D_Classic"]:
    for hpval in (hypervals_div[casename] if hypername=="div" else [0.2, 0.4, 0.6, 0.8]):
        metric_lists =[list() for _ in metric_cats]
        hpvalstr = tostr(hpval)
        try:
            for i in range(0,3):
                f_fb = open("../benchmark_results/hyperparam/"+hypername+"/"+casename+"_"+hpvalstr+"_"+str(i))
                fb_content = f_fb.readline(); f_fb.close()
                fb_content = fb_content.split()
                for j in range(len(metric_cats)):
                    metric_lists[j].append(float(fb_content[j]))
            metric_stats = [(avg(metric_lists[j]),std(metric_lists[j])) for j in range(len(metric_cats))]
            str_buf = [casename, tostr(hpval,False)]
            for j in range(len(metric_cats)):
                m = metric_stats[j]
                str_buf += ["%.6g"%m[0], "%.6g"%m[1]]
            buffer.append(str_buf)
            
        except FileNotFoundError:
            print("results for "+casename+" "+hpvalstr+" not found")

f_csv = open("hyperparam_"+hypername+".csv", 'w')
frow = ["pde", hypername]
for j in range(len(metric_cats)):
    frow += [metric_cats[j]+"_avg", metric_cats[j]+"_std"]
buffer.insert(0, frow)
for entry in buffer:
    f_csv.write(", ".join(entry)+'\n')
f_csv.close()