#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import math
import operator
import sys
from collections import defaultdict, OrderedDict


class Rating(object):
    def __init__(self, user_id, item_id, rating, timestamp):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.rating = int(rating)
        self.timestamp = timestamp

    def __repr__(self):
        return '{} {} {} {}'.format(self.user_id, self.item_id, self.rating, self.timestamp)


class Recommender(object):
    def __init__(self, train_data_filename, test_data_filename):
        self._train_data_filename = train_data_filename
        self._test_data_filename = test_data_filename
        self.users = defaultdict(dict)
        self.items = defaultdict(dict)
        self.user_similarity = defaultdict(dict)

    def _load_ratings(self):
        print("Loading: " + self._train_data_filename)
        with open(self._train_data_filename, 'r') as f:
            for idx, line in enumerate(f):
                # data format [user_id]\t[item_id]\t[rating]\t[time_stamp]\n
                data = line.strip().split('\t')
                r = Rating(data[0], data[1], data[2], data[3])
                self.users[data[0]][data[1]] = r
                self.items[data[1]][data[0]] = r

    def _get_common_item_ids_by_user_ids(self, user_id, opponent_user_id):
        """Returns common item ids between two users"""
        return self.users[user_id].keys() & self.users[opponent_user_id].keys()

    def _calculate_user_similarity(self):
        print("Calculating user similarity scores...")
        for user_id, _ in self.users.items():
            for opponent_user_id, _ in self.users.items():
                if user_id == opponent_user_id:
                    continue

                if self.user_similarity[opponent_user_id].get(user_id):
                    self.user_similarity[user_id][opponent_user_id] = self.user_similarity[opponent_user_id][user_id]
                    continue

                common_item_ids = self._get_common_item_ids_by_user_ids(user_id, opponent_user_id)
                common_items_count = len(common_item_ids)
                if common_items_count == 0:
                    continue

                # Calculate similarity score by using Pearson Correlation Coefficient(PCC)
                user_r_sum = opponent_user_r_sum = 0
                user_r_sq_sum = opponent_user_r_sq_sum = 0
                multiple_sum = 0

                for item_id in common_item_ids:
                    user_rating = self.users[user_id][item_id].rating
                    opponent_user_rating = self.users[opponent_user_id][item_id].rating

                    user_r_sum += user_rating
                    opponent_user_r_sum += opponent_user_rating

                    user_r_sq_sum += user_rating ** 2
                    opponent_user_r_sq_sum += opponent_user_rating ** 2

                    multiple_sum += (user_rating * opponent_user_rating)

                upper_result = multiple_sum - (user_r_sum * opponent_user_r_sum / common_items_count)

                u_sq = user_r_sq_sum - pow(user_r_sum, 2) / common_items_count
                o_u_sq = opponent_user_r_sq_sum - pow(opponent_user_r_sum, 2) / common_items_count
                lower_result = math.sqrt(u_sq * o_u_sq)

                if lower_result == 0 or upper_result == 0:
                    similarity_score = 0
                else:
                    similarity_score = upper_result / lower_result

                self.user_similarity[user_id][opponent_user_id] = similarity_score

    def _predict_rating(self, user_id, item_id):
        similiar_user_objects = self.user_similarity[user_id].items()
        # Order by similarity score (descending order)
        similiar_user_objects = OrderedDict(sorted(similiar_user_objects, key=operator.itemgetter(1), reverse=True))

        upper_result = lower_result = 0
        for similiar_user_id, similiarity_score in similiar_user_objects.items():
            if similiarity_score < 0:
                continue

            similiar_user_item_ids = self.users[similiar_user_id].keys()
            if item_id in similiar_user_item_ids:
                similiar_rating = self.users[similiar_user_id][item_id].rating

                upper_result += (similiar_rating * similiarity_score)
                lower_result += similiarity_score

        if upper_result != 0 and lower_result != 0:
            predicted_rating = round(upper_result / lower_result)
            if predicted_rating < 0:
                return 1
        else:
            return 3

        return predicted_rating

    def _predict(self):
        print("Predicting...")
        name_obj = self._test_data_filename.split('.')[0]
        with open(self._test_data_filename, 'r') as f, open(name_obj + '.base_prediction.txt', 'w') as f2:
            for idx, line in enumerate(f):
                # data format [user_id]\t[item_id]\t[rating]\t[time_stamp]\n
                data = line.strip().split('\t')
                predicted_score = self._predict_rating(data[0], data[1])

                f2.write('{}\t{}\t{}\n'.format(data[0], data[1], predicted_score))

    def run(self):
        self._load_ratings()
        self._calculate_user_similarity()
        self._predict()


if __name__ == '__main__':
    _, train_data_filename, test_data_filename = sys.argv
    recommender = Recommender(train_data_filename, test_data_filename)
    recommender.run()
