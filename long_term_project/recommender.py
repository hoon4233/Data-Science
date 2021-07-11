import sys
from collections import defaultdict
import numpy as np
import time

# TRAINING_DATA_PATH = f"./data/{sys.argv[1]}"
# TEST_DATA_PATH = f"./data/{sys.argv[2]}"
# RESULT_FILE_PATH= f"./test/u{sys.argv[2][1]}.base_prediction.txt"
TRAINING_DATA_PATH = f"./{sys.argv[1]}"
TEST_DATA_PATH = f"./{sys.argv[2]}"
RESULT_FILE_PATH= f"./u{sys.argv[2][1]}.base_prediction.txt"
# print(TRAINING_DATA_PATH,TEST_DATA_PATH,RESULT_FILE_PATH)

class model():
    def __init__(self, TRAINING_DATA_PATH, TEST_DATA_PATH, RESULT_FILE_PATH):
        self.training_set = defaultdict(lambda : defaultdict(float))
        self.test_set = []
        self.RESULT_FILE_PATH = RESULT_FILE_PATH
        with open(TRAINING_DATA_PATH, 'r' ) as train_f :
            trxs = train_f.readlines()
            for trx in trxs :
                trx = trx.strip().split('\t')
                u_id, i_id, rating = trx[0],trx[1],trx[2]
                self.training_set[u_id][i_id] = float(rating)

        with open(TEST_DATA_PATH, 'r') as test_f :
            trxs = test_f.readlines()
            for trx in trxs :
                trx = trx.strip().split('\t')
                u_id, i_id = trx[0],trx[1]
                self.test_set.append((u_id, i_id))

    def pearson_cor_coe(self, me, other) :
        intersection = set()
        for item in self.training_set[me] :
            if item in self.training_set[other] :
                intersection.add(item)
        if intersection :
            mean_1 = np.mean(list(self.training_set[me].values()))
            mean_2 = np.mean(list(self.training_set[other].values()))

            sum_1, ssum_1 = 0,0
            sum_2, ssum_2 = 0,0

            for item in intersection :
                sum_1 += self.training_set[me][item] - mean_1
                ssum_1 += np.power(self.training_set[me][item]-mean_1,2)

            for item in intersection :
                sum_2 += self.training_set[other][item] - mean_2
                ssum_2 += np.power(self.training_set[other][item]-mean_2,2)

            numerator, denominator = sum_1*sum_2, np.sqrt(ssum_1*ssum_2)
            if denominator == 0 : return 0
            else : return numerator/denominator
        return 0

    def make_cf_row(self, me):
        user_item_rating, user_item_sim = defaultdict(int), defaultdict(int)
        for other in self.training_set :
            if me == other : continue
            sim = self.pearson_cor_coe(me, other)
            if sim > 0 :
                for i_id in self.training_set[other]:
                    if i_id not in self.training_set[me] or self.training_set[me][i_id] == 0 :
                        user_item_rating[i_id] += self.training_set[other][i_id] * sim
                        user_item_sim[i_id] += sim

        my_row = {i_id : sigma/user_item_sim[i_id] for i_id, sigma in user_item_rating.items()}
        for i_id, rating in self.training_set[me].items() :
            if i_id not in my_row :
                my_row[i_id] = rating

        self.cf[me] = my_row

    def collaborative_filtering(self):
        self.cf = defaultdict(dict)
        self.result = []
        for trx in self.test_set :
            u_id, i_id = trx[0], trx[1]
            if u_id not in self.cf :
                self.make_cf_row(u_id)
            try :
                self.result.append( (u_id, i_id, self.cf[u_id][i_id] ) )
            except KeyError:
                self.result.append( (u_id, i_id, 3) )

    def save(self) :
        with open(self.RESULT_FILE_PATH, 'w') as f :
            for trx in self.result :
                trx_text = f"{trx[0]}\t{trx[1]}\t{trx[2]}\n"
                f.write(trx_text)

start_time = time.time()

recommender = model(TRAINING_DATA_PATH=TRAINING_DATA_PATH, TEST_DATA_PATH=TEST_DATA_PATH,
                    RESULT_FILE_PATH=RESULT_FILE_PATH)
recommender.collaborative_filtering()
recommender.save()

print(f"Execution time : {time.time()-start_time}")