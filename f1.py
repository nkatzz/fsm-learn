def precision(tps, fps):
    return float(tps)/(tps+fps)

def recall(tps, fns):
    return float(tps) / (tps + fns)

def f1(tps, fps, fns):
    p = precision(tps, fps)
    r = recall(tps, fns)
    return 2*p*r/(p+r)

print(f1(18,1,2))