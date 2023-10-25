class ClusterOfWordsEvaluator:
    positive_label = True  # essay grade is 40 (level 1)
    negative_label = False  # essay grade is >= 80 (level >= 2)

    def __init__(self, min_perplexity: float):
        self.min_perplexity = min_perplexity

    def evaluate(self, essay_perplexity: float):
        if self._essay_is_a_cluster_of_words(essay_perplexity):
            return self.positive_label
        else:
            return self.negative_label

    def _essay_is_a_cluster_of_words(self, essay_perplexity: float):
        return essay_perplexity >= self.min_perplexity
