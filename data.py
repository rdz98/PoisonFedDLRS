import numpy as np


def load_file(file_path):
    m_item, all_pos = 0, []

    with open(file_path, "r") as f:
        for line in f.readlines():
            pos = list(map(int, line.rstrip().split(' ')))[1:]
            if pos:
                m_item = max(m_item, max(pos) + 1)
            all_pos.append(pos)

    return m_item, all_pos


def load_dataset(path):
    m_item = 0
    m_item_, all_train_ind = load_file(path + "/train.dat")
    m_item = max(m_item, m_item_)
    m_item_, all_test_ind = load_file(path + "/test.dat")
    m_item = max(m_item, m_item_)

    items_popularity = np.zeros(m_item)
    for items in all_train_ind:
        for item in items:
            items_popularity[item] += 1
    for items in all_test_ind:
        for item in items:
            items_popularity[item] += 1

    return m_item, all_train_ind, all_test_ind, items_popularity
