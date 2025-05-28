from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from MusicRecommendation import MusicRecommendation
from RBM import RBM
import pandas as pd

class RBMAlgorithm(AlgoBase):

    def __init__(self, epochs=20, hiddenDim=100, learningRate=0.001, batchSize=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.musicRecommendation = MusicRecommendation()
        self.musicRecommendation.loadMusicData()
        self.stoplist = ["sex", "drugs", "rock n roll"]

    def buildStoplist(self, trainset):
        self.stoplistLookup = {}
        for iid in trainset.all_items():
            self.stoplistLookup[iid] = False
            musicID = trainset.to_raw_iid(iid)
            musicName = self.musicRecommendation.getMusicName(musicID)
            if musicName:
                musicName = musicName.lower()
                for term in self.stoplist:
                    if term in musicName:
                        print("Blocked ", musicName)
                        self.stoplistLookup[iid] = True

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def _calculate_quantile_thresholds(self, trainset, num_levels=10):
        """
        Tính toán các ngưỡng điểm để chia dữ liệu thành N cấp độ.
        """
        print("Calculating quantile thresholds for rating normalization...")
        # Lấy tất cả các điểm ratings
        all_ratings = [r for (_, _, r) in trainset.all_ratings()]
        
        # Sử dụng Pandas để dễ dàng tính toán quantiles
        ratings_series = pd.Series(all_ratings)
        
        # Tính toán các ngưỡng. Ví dụ, cho 10 cấp độ, chúng ta cần các ngưỡng 0.1, 0.2, ..., 0.9
        quantiles = np.linspace(0, 1, num_levels + 1)[1:-1] # -> [0.1, 0.2, ..., 0.9]
        
        # Lấy các giá trị điểm tại các ngưỡng đó
        thresholds = ratings_series.quantile(quantiles).tolist()
        print(f"Calculated thresholds for {num_levels} levels: {thresholds}")
        
        self.rating_thresholds = thresholds
        return thresholds

    def _normalize_rating(self, rating):
        """
        Chuyển đổi một điểm rating thô thành một cấp độ (0-9) dựa trên ngưỡng.
        """
        # np.searchsorted tìm vị trí mà rating nên được chèn vào mảng thresholds để duy trì trật tự.
        # Kết quả chính là cấp độ rating của chúng ta.
        # Ví dụ: nếu rating nhỏ hơn ngưỡng đầu tiên, kết quả là 0.
        # Nếu rating lớn hơn ngưỡng cuối cùng, kết quả là 9.
        return np.searchsorted(self.rating_thresholds, rating)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        self.buildStoplist(trainset)

        self._calculate_quantile_thresholds(trainset, num_levels=10)

        numUsers = trainset.n_users
        numItems = trainset.n_items
        
        trainingMatrix = np.zeros([numUsers, numItems, 10], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            # if not self.stoplistLookup[iid]: # Bỏ qua stoplist để ví dụ đơn giản
                # Chuẩn hóa rating thành một số từ 0-9
                normalized_level = self._normalize_rating(rating)
                
                # Gán vào ma trận training
                trainingMatrix[int(uid), int(iid), normalized_level] = 1
        
        # Flatten to a 2D array, with nodes for each possible rating type on each possible item, for every user.
        trainingMatrix = np.reshape(trainingMatrix, [trainingMatrix.shape[0], -1])
        
        # Create an RBM with (num items * rating values) visible nodes
        rbm = RBM(trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim, learningRate=self.learningRate, batchSize=self.batchSize, epochs=self.epochs)
        rbm.Train(trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        for uiid in range(trainset.n_users):
            if (uiid % 50 == 0):
                print("Processing user ", uiid)
            recs = rbm.GetRecommendations([trainingMatrix[uiid]])
            recs = np.reshape(recs, [numItems, 10])
            
            for itemID, rec in enumerate(recs):
                normalized = self.softmax(rec)
                rating = np.average(np.arange(10), weights=normalized)
                self.predictedRatings[uiid, itemID] = (rating + 1) * 0.5
        
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        
        rating = self.predictedRatings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating
