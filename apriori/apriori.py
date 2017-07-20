#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import itertools
import sys
from decimal import Decimal, ROUND_HALF_UP


class Apriori(object):
    def __init__(self, min_support, input_filename, output_filename):
        self._freq_dict = {}
        self._total_transaction_count = 0
        self._min_support = int(min_support)
        self._input_filename = input_filename
        self._output_filename = output_filename

    def _ready(self):
        print('Ready...')
        with open(self._input_filename, 'r') as f:
            line_count = 0
            for line in f:
                line_count += 1
                # Convert data into set of numbers
                numbers = set(map(int, line.strip().split('\t')))

                for number in numbers:
                    key = frozenset([number])
                    if not self._freq_dict.get(key):
                        data = {
                            'freq': 1,
                            'support': 0,
                            'itemsets': []
                        }
                        self._freq_dict[key] = data
                    else:
                        updated_freq = self._freq_dict[key]['freq'] + 1
                        data = {
                            'freq': updated_freq,
                            'support': updated_freq / line_count * 100
                        }
                        self._freq_dict[key].update(data)

            self._total_transaction_count = line_count

    def _generate_candidates(self, itemset_length):
        candidates = []
        for k in self._freq_dict.keys():
            if len(k) != (itemset_length - 1):
                continue

            if self._freq_dict[k]['support'] >= self._min_support:
                candidates.append(k)

        candidates = frozenset().union(*candidates)
        combinations = itertools.combinations(candidates, itemset_length)

        # Convert tuples into set list
        new_candidates = []
        for k, t in enumerate(combinations):
            new_candidates.append(frozenset(t))

        del combinations

        return new_candidates

    def _apriori(self):
        iteration = 1

        while True:
            iteration += 1

            print('Run apriori iter#', iteration)
            candidates = self._generate_candidates(iteration)
            print('Candidates: ', len(candidates))
            if not candidates:
                break

            with open(self._input_filename, 'r') as f:
                for n, line in enumerate(f):
                    numbers = frozenset(map(int, line.strip().split('\t')))

                    if len(numbers) < iteration:
                        continue

                    for candidate in candidates:
                        if candidate.issubset(numbers):
                            if not self._freq_dict.get(candidate):
                                data = {
                                    'freq': 1,
                                    'support': 0,
                                    'itemsets': []
                                }
                                self._freq_dict[candidate] = data
                            else:
                                updated_freq = self._freq_dict[candidate]['freq'] + 1
                                data = {
                                    'freq': updated_freq,
                                    'support': updated_freq / self._total_transaction_count * 100
                                }
                                self._freq_dict[candidate].update(data)

                            del data

                            for k in self._freq_dict.keys():
                                if k.issubset(candidate) and k != candidate:
                                    self._freq_dict[k]['itemsets'].append(candidate)

                    del numbers

    def _print(self):
        with open(self._output_filename, 'w') as f:
            for freq, v in self._freq_dict.items():
                for itemset in set(v['itemsets']):
                    association = itemset - freq

                    formatted_support = self._format_float_number(self._freq_dict[itemset]['support'])
                    confidence = self._freq_dict[itemset]['freq'] / v['freq'] * 100
                    formatted_confidence = self._format_float_number(confidence)

                    if formatted_support >= self._min_support:
                        f.write('{}\t{}\t{:.2f}\t{:.2f}'.format(
                            self._format_itemsets(freq),
                            self._format_itemsets(association),
                            formatted_support,
                            formatted_confidence
                        ) + '\n')

    @staticmethod
    def _format_itemsets(itemset):
        # To show itemset items by ascending order
        new_itemset = list(itemset)
        new_itemset.sort()
        return '{{{}}}'.format(','.join(str(x) for x in new_itemset))

    @staticmethod
    def _format_float_number(number):
        return Decimal(number).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)

    def run(self):
        # Calculate total transaction count and make a base length-1 itemsets
        self._ready()
        self._apriori()
        self._print()

if __name__ == '__main__':
    _, min_support, input_filename, output_filename = sys.argv
    apriori = Apriori(min_support, input_filename, output_filename)
    apriori.run()
