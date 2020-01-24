import csv
import sys
import json
import math
import time
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


user_list, business_list = list(), list()
user_table, business_table = dict(), dict()
user_dict, business_dict = dict(), dict()
user_norm, business_norm = dict(), dict()
user_sum, business_sum = dict(), dict()
business_stars, business_vector = dict(), dict()
pair_weights = dict()

train_file = sys.argv[1]
feature_file = sys.argv[2]
test_file = sys.argv[3]
case_number = int(sys.argv[4])
output_file = sys.argv[5]
DEBUG = False

def get_user_id(user):
    if not user in user_table:
        user_list.append(user)
        user_table[user] = len(user_list)-1
    return user_table[user]

def get_business_id(business):
    if not business in business_table:
        business_list.append(business)
        business_table[business] = len(business_list)-1
    return business_table[business]

def getPairs(ids):
    ids.sort()
    l = len(ids)
    ret = list()
    for i in range(l-1):
        for j in range(i+1, l):
            ret.append((ids[i], ids[j]))
    return ret

def pearsonCorrelation(pair):
    user1, user2 = pair[0], pair[1]
    if not user1 in user_dict or not user2 in user_dict:
        return 0
    rate1, rate2 = list(), list()
    for business in user_dict[user1]:
        if business in user_dict[user2]:
            rate1.append(user_dict[user1][business])
            rate2.append(user_dict[user2][business])
    if len(rate1) == 0:
        return 0
    avg1, avg2 = sum(rate1)/len(rate1), sum(rate2)/len(rate2)
    p, sum1, sum2 = 0, 0, 0
    for i in range(len(rate1)):
        p += ((rate1[i]-avg1)*(rate2[i]-avg2))
        sum1 += ((rate1[i]-avg1)**2)
        sum2 += ((rate2[i]-avg2)**2)
    weight = 0 if p == 0 else p / (sum1*sum2)**0.5
    return weight

def pearsonCorrelationItems(pair):
    bus1, bus2 = pair[0], pair[1]
    if not bus1 in business_dict or not bus2 in business_dict:
        return 0
    rate1, rate2 = list(), list()
    for user in business_dict[bus1].keys():
        if user in business_dict[bus2]:
            rate1.append(business_dict[bus1][user])
            rate2.append(business_dict[bus2][user])
    if len(rate1) == 0:
        return 0
    avg1, avg2 = sum(rate1)/len(rate1), sum(rate2)/len(rate2)
    p, sum1, sum2 = 0, 0, 0
    for i in range(len(rate1)):
        p += ((rate1[i]-avg1)*(rate2[i]-avg2))
        sum1 += ((rate1[i]-avg1)**2)
        sum2 += ((rate2[i]-avg2)**2)
    weight = 0 if p == 0 else p / (sum1*sum2)**0.5
    return weight

def cosineSimilarity(pair, record, norm):
    key1, key2 = pair[0], pair[1]
    if not key1 in record or not key2 in record:
        return 0
    p = 0
    for candidate in record[key1]:
        if candidate in record[key2]:
            p += (record[key1][candidate]*record[key2][candidate])
    return p / (norm[key1]*norm[key2])**0.5

def contentSimilarity(pair):
    bus1, bus2 = pair[0], pair[1]
    if not bus1 in business_vector or not bus2 in business_vector:
        return 0
    a1, a2 = business_vector[bus1][0], business_vector[bus1][1]
    b1, b2 = business_vector[bus2][0], business_vector[bus2][1]
    p = a1*b1 + a2*b2
    q = ((a1**2+a2**2)*(b1**2+b2**2)) ** 0.5
    return p/q

def predictUserBased(target):
    user, business = target[0], target[1]
    if user in user_dict and business in user_dict[user]:
        return (target, user_dict[user][business])
    elif not user in user_dict:
        if business in business_stars:
            return (target, business_stars[business])
        else:
            return (target, 3.0)
    else:
        user_avg = user_sum[user] / len(user_dict[user])
        if not business in business_dict:
            return (target, user_avg)
        else:
            p, q = 0, 0
            for other in business_dict[business].keys():
                pair = (user, other) if user < other else (other, user)
                if not pair in pair_weights:
                    pair_weights[pair] = cosineSimilarity(pair, user_dict, user_norm)
                    #pair_weights[pair] = pearsonCorrelation(pair)
                if pair_weights[pair] > 0:
                    other_avg = (user_sum[other]-user_dict[other][business]) / (len(user_dict[other])-1)
                    p += ((user_dict[other][business]-other_avg)*pair_weights[pair])
                    q += abs(pair_weights[pair])
            if q == 0:
                prediction = user_avg
            else:
                prediction = user_avg+p/q
                prediction = max(1.0, prediction)
                prediction = min(5.0, prediction)
            return (target, prediction)

def predictItemBased(target):
    user, business = target[0], target[1]
    if user in user_dict and business in user_dict[user]:
        return (target, user_dict[user][business])
    elif not business in business_dict:
        if business in business_stars:
            return (target, business_stars[business])
        else:
            return (target, 3.0)
    else:
        if business in business_stars:
            business_avg = business_stars[business]
        else:
            business_avg = business_sum[business] / len(business_dict[business])
        if not user in user_dict:
            return (target, business_avg)
        else:
            p, q = 0, 0
            for other in user_dict[user].keys():
                pair = (business, other) if business < other else (other, business)
                if not pair in pair_weights:
                    pair_weights[pair] = cosineSimilarity(pair, business_dict, business_norm)
                if pair_weights[pair] != 0:
                    if other in business_stars:
                        other_avg = business_stars[other]
                    else:
                        other_avg = business_sum[other] / len(business_dict[other])
                    p += ((business_dict[other][user]-other_avg)*pair_weights[pair])
                    q += abs(pair_weights[pair])
            if q == 0 or p == 0:
                prediction = business_avg
            else:
                prediction = business_avg + p/q
                prediction = max(1.0, prediction)
                prediction = min(5.0, prediction)
            return (target, prediction)

def predictItemBasedHybrid(target):
    user, business = target[0], target[1]
    if user in user_dict and business in user_dict[user]:
        return (target, user_dict[user][business])
    elif not business in business_dict and not business in business_stars:
        return (target, 3.5)
    else:
        if business in business_stars:
            business_avg = business_stars[business]
        else:
            business_avg = business_sum[business] / len(business_dict[business])
        if not user in user_dict:
            return (target, business_avg)
        else:
            p, q = 0, 0
            for other in user_dict[user].keys():
                pair = (business, other) if business < other else (other, business)
                if not pair in pair_weights:
                    cos_weight = cosineSimilarity(pair, business_dict, business_norm)
                    content_weight = contentSimilarity(pair)
                    pair_weights[pair] = cos_weight if cos_weight > 0 else content_weight
                if pair_weights[pair] > 0:
                    if other in business_stars:
                        other_avg = business_stars[other]
                    else:
                        other_avg = business_sum[other] / len(business_dict[other])
                    p += ((business_dict[other][user]-other_avg)*pair_weights[pair])
                    q += abs(pair_weights[pair])
            if q == 0 or p == 0:
                prediction = business_avg
            else:
                prediction = business_avg + p/q
                prediction = max(1.0, prediction)
                prediction = min(5.0, prediction)
            return (target, prediction)


start = time.time()
sc = SparkContext(master='local[*]', appName='inf553_hw3_task2')
train_data = sc.textFile(train_file, 8).filter(lambda x: x != 'user_id, business_id, stars') \
    .map(lambda x: x.split(','))
business_stars = sc.textFile(feature_file, 8) \
    .map(lambda x: (json.loads(x)['business_id'], json.loads(x)['stars'])).collectAsMap()

user_list = train_data.map(lambda x: x[0]).distinct().collect()
business_list = train_data.map(lambda x: x[1]).distinct().collect()
for index, user in enumerate(user_list):
    user_table[user] = index
for index, business in enumerate(business_list):
    business_table[business] = index
for rating in train_data.collect():
    user_id, business_id, stars = rating[0], rating[1], float(rating[2])
    if user_id in user_dict:
        user_dict[user_id][business_id] = stars
        user_norm[user_id] += stars**2
        user_sum[user_id] += stars
    else:
        user_dict[user_id] = {business_id: stars}
        user_norm[user_id] = stars**2
        user_sum[user_id] = stars
    if business_id in business_dict:
        business_dict[business_id][user_id] = stars
        business_norm[business_id] += stars**2
        business_sum[business_id] += stars
    else:
        business_dict[business_id] = {user_id: stars}
        business_norm[business_id] = stars**2
        business_sum[business_id] = stars

test_data = sc.textFile(test_file, 8).filter(lambda x: x != 'user_id, business_id, stars') \
    .map(lambda x: x.split(','))

if case_number == 1:
    ratings = train_data.map(lambda x: Rating(user_table[x[0]], business_table[x[1]], float(x[2])))
    model = ALS.train(ratings=ratings, rank=2, iterations=10, nonnegative=True)
    test_input = test_data.map(lambda x: (get_user_id(x[0]), get_business_id(x[1])))
    predictions = model.predictAll(test_input) \
        .map(lambda x: ((user_list[x[0]], business_list[x[1]]), float(x[2])))
if case_number == 2:
    predictions = test_data.map(lambda x: (x[0], x[1])).map(predictUserBased)
if case_number == 3:
    predictions = test_data.map(lambda x: (x[0], x[1])).map(predictItemBased)
if case_number == 4:
    max_review_nums = sc.textFile(feature_file, 8).map(lambda x: json.loads(x)['review_count']).max()
    business_vector = sc.textFile(feature_file, 8).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['business_id'], (x['stars']/5.0, x['review_count']/max_review_nums))) \
        .collectAsMap()
    predictions = test_data.map(lambda x: (x[0], x[1])).map(predictItemBasedHybrid)

if DEBUG:
    ratesAndPreds = test_data.map(lambda x: ((x[0], x[1]), float(x[2]))) \
        .join(predictions)
    RMSE = math.sqrt(ratesAndPreds.map(lambda x: (x[1][0] - x[1][1])**2).mean())
    print("Root Mean Squared Error = " + str(RMSE))
    print(time.time()-start)

fout = open(output_file, mode='w')
fwriter = csv.writer(fout, delimiter=',', quoting=csv.QUOTE_MINIMAL)
fwriter.writerow(['user_id', ' business_id', ' prediction'])
for pair in predictions.collect():
    fwriter.writerow([str(pair[0][0]), str(pair[0][1]), pair[1]])
fout.close()