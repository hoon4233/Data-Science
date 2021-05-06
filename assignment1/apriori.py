import sys
from itertools import chain, combinations

min_support = float(sys.argv[1]) / 100
input_file, output_file = sys.argv[2], sys.argv[3]

def apriori(trxs):
    global min_support
    trxs_len = len(trxs)
    
    idxs = set([])
    for trx in trxs :
        idxs = idxs.union(trx)

    # 일반 set 은 unhashable type 이므로 frozenset 사용
    candidates = { frozenset({i}) for i in idxs }
    # candidates = { set({i}) for i in idxs }

    item_set = dict()
    K = 1

    while candidates :
        count = dict()

        for trx in trxs :
            for candidate in candidates:
                if candidate.issubset(trx):
                    try :
                        count[candidate] += 1
                    except KeyError :
                        count[candidate] = 1

        # pruning
        after_pruning = { key : (float(value) / trxs_len) for (key, value) in count.items() if (float(value) / trxs_len) >= min_support }
        item_set[K] = after_pruning

        #self_joining
        K += 1
        candidates = { i.union(j) for i in after_pruning for j in after_pruning if len(i.union(j)) == K }


    return item_set

def print_output(trxs, fps):
    for patt_len, patt_len_fps in fps.items():
        if patt_len == 1 :
         continue
        for fp in patt_len_fps :
            com_len_cases = [ combinations(fp, length) for length in range(1, len(fp)+1, 1) ]
            all_cases = []
            for  cases in com_len_cases :
                for case in cases :
                    all_cases.append(frozenset(case))

            for case in all_cases:
                remainder = fp.difference(case)

                if remainder :
                    confidence = fps[len(fp)][fp] / fps[len(case)][case]
                    prt_case, prt_remainder = str(set(map(int,case))).replace(" ", ""), str(set(map(int,remainder))).replace(" ", "")
                    prt_supp, prt_confi =  str('%.2f' % round(fps[len(fp)][fp] * 100, 2)), str('%.2f' % round(confidence * 100, 2))
                    string = prt_case + '\t' + prt_remainder + '\t' + prt_supp + '\t' + prt_confi + '\n'

                    with open(output_file, 'a') as f :
                        f.write(string)
            

        


with open(input_file, 'r') as f :
    trxs = [ trx.split('\t') for trx in f.read().splitlines()  ]

    fps = apriori(trxs)
    print_output(trxs, fps)