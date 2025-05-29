from MusicRecommendation import MusicRecommendation
from RBMAlgorithm import RBMAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator
import random
import numpy as np


def LoadMusicsData():
    musicData = MusicRecommendation()
    data = musicData.loadMusicData()
    rankings = musicData.getPopularityRanks()
    users = musicData.loadListeners()
    return (musicData, data, rankings, users)
    

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(musicData, evaluationData, rankings, users) = LoadMusicsData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#Simple RBM
SimpleRBM = RBMAlgorithm(epochs=40)
#Content
ContentKNN = ContentKNNAlgorithm(10, {}, musicData)

#Combine
Hybrid = HybridAlgorithm([SimpleRBM, ContentKNN], [0.2, 0.8])


evaluator.AddAlgorithm(Hybrid, "Hybrid")


recommendForEveryUser = evaluator.RecommendForEachUser(musicData, users)


# Save recommend course ids to course_recommendations table
musicData.saveAllRecommendationsToRedis(recommendForEveryUser)
