#!/usr/bin/env python3
from __future__ import division

import math
import sys
from collections import Counter, OrderedDict


class Node(object):
    pass


class AttributeNode(Node):
    def __init__(self, name):
        super(Node, self).__init__()
        self.name = name
        self.criteria = []

    def __repr__(self):
        return str(self.__dict__)


class CriteriaNode(Node):
    def __init__(self, name, data_count):
        super(Node, self).__init__()
        self.name = name
        self.data_count = data_count
        self.attribute_node = None
        self.class_node = None

    def __repr__(self):
        return str(self.__dict__)


class ClassNode(Node):
    def __init__(self, value):
        super(Node, self).__init__()
        self.value = value

    def __repr__(self):
        return self.value


class DecisionTreeBuilder(object):
    def __init__(self, train_set_filename, test_set_filename, output_filename):
        self.train_set_filename = train_set_filename
        self.test_set_filename = test_set_filename
        self.output_filename = output_filename
        self.initial_train_set = []
        self.initial_attributes = []

        self._load_data_set()

    def _load_data_set(self):
        with open(self.train_set_filename, 'r') as f:
            for idx, line in enumerate(f):
                data = line.strip().split('\t')
                if idx == 0:
                    self.initial_attributes = data
                else:
                    self.initial_train_set.append(data)

    @staticmethod
    def _calculate_info_value(frequency_list):
        total_count = sum(frequency_list)
        info_value = 0
        for f in frequency_list:
            value = (f / total_count)
            info_value -= (value * math.log(value, 2))

        return info_value

    @staticmethod
    def _get_info_d_value(train_set):
        class_data = [x[-1] for x in train_set]
        class_data.sort()

        c = Counter(class_data)

        return DecisionTreeBuilder._calculate_info_value(c.values())

    @staticmethod
    def _calculate_information_gains(train_set, attributes, info_d):
        # Get all attributes without class attribute
        attr_gains = {}
        for attr in attributes:
            attr_idx = attributes.index(attr)

            # find a distinct attributes for attribute from train_set
            attr_data = [x[attr_idx] for x in train_set]
            criteria_c = Counter(attr_data)

            info_attr = 0
            # Loop each criteria by distinct attribute data
            for idx, criteria in enumerate(criteria_c.keys()):
                class_list = []
                for data in train_set:
                    # To get class value from criteria in attr
                    if data[attr_idx] == criteria:
                        class_list.append(data[-1])

                c = Counter(class_list)
                count = list(criteria_c.values())[idx]
                value = (count / len(attr_data))

                info_attr += (value * DecisionTreeBuilder._calculate_info_value(c.values()))

            attr_gains[attr] = (info_d - info_attr)

        # Return attributes with descending information gain order
        return OrderedDict(sorted(attr_gains.items(), key=lambda d: d[1], reverse=True))

    @staticmethod
    def _get_criteria(train_set, attr_idx):
        attr_data = [x[attr_idx] for x in train_set]

        criteria_c = Counter(attr_data)
        return criteria_c.keys()

    @staticmethod
    def _select_attribute(train_set, attributes):
        info_d = DecisionTreeBuilder._get_info_d_value(train_set)
        # Ignore class attr
        attr_gains = DecisionTreeBuilder._calculate_information_gains(train_set, attributes[:-1], info_d)

        return list(attr_gains.items())[0]

    @staticmethod
    def _split_by_criteria(train_set, attr_idx, criteria):
        split_data = []
        for x in train_set:
            if x[attr_idx] == criteria:
                new_data = x[:attr_idx] + x[attr_idx+1:]
                split_data.append(new_data)

        return split_data

    @staticmethod
    def _get_majority_class(train_set):
        train_set_to_check = [x[-1] for x in train_set]
        train_set_to_check.sort()

        c = Counter(train_set_to_check)
        return c.most_common(1)[0][0]

    @staticmethod
    def _build_tree(train_set, attributes):
        class_data = [x[-1] for x in train_set]
        # If all class label in train set is same
        if len(set(class_data)) == 1:
            # Return class label
            return ClassNode(class_data[-1])

        # If there's no attribute to examine
        if len(attributes) < 3:
            # Return class label majority in train_set
            major_class = DecisionTreeBuilder._get_majority_class(train_set)
            return ClassNode(major_class)

        selected_attribute, info_gain = DecisionTreeBuilder._select_attribute(train_set, attributes)

        attr_idx = attributes.index(selected_attribute)
        reduced_attributes = attributes[:]
        reduced_attributes.remove(selected_attribute)

        attribute_node = AttributeNode(selected_attribute)
        for criteria in DecisionTreeBuilder._get_criteria(train_set, attr_idx):
            split_train_set = DecisionTreeBuilder._split_by_criteria(train_set, attr_idx, criteria)

            criteria_node = CriteriaNode(criteria, len(split_train_set))
            attribute_node.criteria.append(criteria_node)
            if len(split_train_set) > 0:
                node = DecisionTreeBuilder._build_tree(split_train_set, reduced_attributes)
                if node is not None:
                    if isinstance(node, AttributeNode):
                        criteria_node.attribute_node = node
                    elif isinstance(node, ClassNode):
                        criteria_node.class_node = node
            else:
                # Choose major class label from the parent's train set if split train set is empty
                major_class = DecisionTreeBuilder._get_majority_class(train_set)
                criteria_node.class_node = ClassNode(major_class)

        # Return root node
        return attribute_node

    @staticmethod
    def _get_class_using_tree(node, data_row, attributes):
        if isinstance(node, ClassNode):
            return node.value

        if isinstance(node, CriteriaNode):
            return node.class_node.value

        if isinstance(node, AttributeNode):
            attr_idx = attributes.index(node.name)
            max_count = 0
            max_count_criteria_name = None
            for criteria in node.criteria:
                if criteria.data_count > max_count:
                    max_count = criteria.data_count
                    max_count_criteria_name = criteria.name

                if data_row[attr_idx] == criteria.name:
                    if criteria.class_node is not None:
                        return DecisionTreeBuilder._get_class_using_tree(
                            criteria.class_node,
                            data_row,
                            attributes
                        )
                    elif criteria.attribute_node is not None:
                        return DecisionTreeBuilder._get_class_using_tree(
                            criteria.attribute_node,
                            data_row,
                            attributes
                        )

            # Choose the criteria name from bigger train set if there's no matching criteria
            if max_count_criteria_name is not None:
                data_row[attr_idx] = max_count_criteria_name

            return DecisionTreeBuilder._get_class_using_tree(node, data_row, attributes)

    def classify(self, tree):
        result = attributes = []
        with open(self.test_set_filename, 'r') as f:
            for idx, line in enumerate(f):
                data = line.strip().split('\t')
                if idx == 0:
                    attributes = data
                    continue

                class_result = DecisionTreeBuilder._get_class_using_tree(tree, data, attributes)
                data.append(class_result)

                result.append('\t'.join(data))

        with open(self.output_filename, 'w') as f:
            f.write('\t'.join(attributes) + '\tClass' + '\n')
            for line in result:
                f.write(line + '\n')

    def run(self):
        decision_tree = self._build_tree(self.initial_train_set, self.initial_attributes)
        self.classify(decision_tree)


if __name__ == '__main__':
    _, train_set_filename, test_set_filename, output_filename = sys.argv
    builder = DecisionTreeBuilder(train_set_filename, test_set_filename, output_filename)
    builder.run()
