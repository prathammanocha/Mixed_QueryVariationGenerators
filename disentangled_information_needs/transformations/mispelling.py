from textattack.augmentation import Augmenter
from textattack.transformations import *
from textattack.constraints.pre_transformation import StopwordModification
from tqdm import tqdm
import logging

CONSTRAINTS = [StopwordModification()]

class MispellingActions():
  def init (self, queries, q_ids):
    self.queries = queries
    self.q_ids = q_ids
    self.augmenters = [
    Augmenter(transformation=WordSwapNeighboringCharacterSwap(), transformations_per_example=1, constraints=CONSTRAINTS),
    Augmenter(transformation=WordSwapRandomCharacterSubstitution(), transformations_per_example=1, constraints=CONSTRAINTS),
    Augmenter(transformation=WordSwapQWERTY(), transformations_per_example=1, constraints=CONSTRAINTS),
    ]

def mispelling_chars(self, specific_queries=None, specific_q_ids=None, sample=None):
  logging.info("Adding mispelling errors using texttattack.")
  logging.info("Methods used: {}.".format(str([t.transformation.__class__.__name__ for t in self.augmenters])))

  if specific_queries:
      queries_to_use = specific_queries
      q_ids_to_use = specific_q_ids
  else:
      queries_to_use = self.queries
      q_ids_to_use = self.q_ids

  query_variations = []
  for i, query in enumerate(tqdm(queries_to_use)):
      for augmenter in self.augmenters:
          try:
              augmented = augmenter.augment(query)
          except: #empty error for QWERTY.
              augmented = [query]
          if not augmented:  # Check if augmented is empty or None
              print(f"No augmentation for query: {query}")
              continue
          for q_variation in augmented:
              query_variations.append([q_ids_to_use[i], query, q_variation, augmenter.transformation.__class__.__name__, "mispelling"])

      if sample and i+1 > sample:
          break

  # Return the query_variations
  return query_variations
