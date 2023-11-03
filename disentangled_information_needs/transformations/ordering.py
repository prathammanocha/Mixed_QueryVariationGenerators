from textattack.augmentation import Augmenter
from textattack.transformations import *
from IPython import embed
from tqdm import tqdm

import pandas as pd
import logging 

class OrderingActions():
    def __init__(self, queries, q_ids):
        self.queries = queries
        self.q_ids = q_ids
        self.augmenters = [
            Augmenter(transformation=WordInnerSwapRandom(), transformations_per_example=1)            
        ]

    def shuffle_word_order(self, specific_queries=None, specific_q_ids=None, sample=None):
        logging.info("Shuffling order of the words using texttattack.")
        logging.info("Methods used: {}.".format(str([t.transformation.__class__.__name__ for t in self.augmenters])))
        
        if specific_queries is None:
            specific_queries = self.queries
        if specific_q_ids is None:
            specific_q_ids = self.q_ids

        i = 0
        query_variations = []
        for query, q_id in zip(specific_queries, specific_q_ids):
            for augmenter in self.augmenters:
                augmented = augmenter.augment(query)
                for q_variation in augmented:
                    query_variations.append([q_id, query, q_variation, augmenter.transformation.__class__.__name__, "ordering"])
            i += 1
            if sample and i > sample:
                break
        return query_variations