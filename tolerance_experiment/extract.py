f = open("./result.txt", "r")

results = {}
current_key = ''

while True:
    try:    
        line = f.readline()
        sline = line.split(',')
        if 'tolerance' in line:
            current_key = sline[1]
            results[current_key] = { 'accept': 0, 'false_positive': 0, 'false_negative': 0 }
            continue
        if 'train/n0' in sline[0] and 'unknown_person' in sline[1]:
            results[current_key]['accept'] += 1
            continue
        if 'train/n0' in sline[0] and 'no_persons_found' in sline[1]:
            results[current_key]['false_negative'] += 1
            continue
        if 'train/n0' in sline[0]:
            results[current_key]['false_positive'] += 1
            continue
        if sline[1] in sline[0]:
            results[current_key]['accept'] += 1
            continue
        else:
            results[current_key]['false_negative'] += 1
            continue
    except:
        break

r = open('extrated.csv', 'a')
for key, value in results.items():
    r.write('{},{},{},{}'.format(value['accept'], value['false_positive'],value['false_negative'], key))

r.close()
