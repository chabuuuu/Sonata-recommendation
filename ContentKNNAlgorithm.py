from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase):

    def __init__(self, k=40, sim_options={}, musicRecommendation=None):
        super().__init__()
        self.k = k
        self.musicRecommendation = musicRecommendation

    def fit(self, trainset):
        super().fit(trainset)

        # Compute item similarity matrix based on music attributes
        print("Computing content-based similarity matrix...")

        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for thisRating in range(self.trainset.n_items):
            if thisRating % 100 == 0:
                print(thisRating, " of ", self.trainset.n_items)

            for otherRating in range(thisRating + 1, self.trainset.n_items):
                thisMusicID = self.trainset.to_raw_iid(thisRating)
                otherMusicID = self.trainset.to_raw_iid(otherRating)

                # Calculate similarity based on custom attributes
                similarity = self.computeSimilarity(thisMusicID, otherMusicID)
                self.similarities[thisRating, otherRating] = similarity
                self.similarities[otherRating, thisRating] = similarity

        print("...done.")
        return self

    def computeSimilarity(self, music_id1, music_id2):
        """
        Tính toán độ tương đồng giữa hai bài hát dựa trên các thuộc tính của chúng.
        Độ tương đồng được tính bằng tổng có trọng số của sự tương đồng trên từng thuộc tính.
        - Đối với các thuộc tính đơn giá trị (quốc gia, người đóng góp), kiểm tra sự trùng khớp.
        - Đối với các thuộc tính đa giá trị (thể loại, nghệ sĩ), sử dụng Chỉ số Jaccard.

        Args:
            music_id1 (int): ID của bài hát thứ nhất.
            music_id2 (int): ID của bài hát thứ hai.

        Returns:
            float: Một điểm tương đồng từ 0.0 đến 1.0.
        """
        # Lấy thông tin chi tiết của hai bài hát từ dữ liệu đã tải
        details1 = self.musicRecommendation.musicID_to_details.get(music_id1, {})
        details2 = self.musicRecommendation.musicID_to_details.get(music_id2, {})

        if not details1 or not details2:
            return 0.0 # Trả về 0 nếu một trong hai bài hát không có thông tin

        # --- Định nghĩa trọng số cho từng thuộc tính (tổng bằng 1.0) ---
        weights = {
            'genres': 0.4,
            'artists': 0.3,
            'categories': 0.1,
            'periods': 0.1,
            'nationality': 0.05,
            'contributor': 0.05
        }

        total_similarity = 0.0

        # 1. Tương đồng về THỂ LOẠI (sử dụng Jaccard)
        genres1 = set(details1.get('genre_ids', []))
        genres2 = set(details2.get('genre_ids', []))
        if genres1 or genres2:
            intersection_genres = len(genres1.intersection(genres2))
            union_genres = len(genres1.union(genres2))
            jaccard_genres = intersection_genres / union_genres if union_genres > 0 else 0
            total_similarity += weights['genres'] * jaccard_genres

        # 2. Tương đồng về NGHỆ SĨ (sử dụng Jaccard)
        artists1 = set(details1.get('artist_ids', []))
        artists2 = set(details2.get('artist_ids', []))
        if artists1 or artists2:
            intersection_artists = len(artists1.intersection(artists2))
            union_artists = len(artists1.union(artists2))
            jaccard_artists = intersection_artists / union_artists if union_artists > 0 else 0
            total_similarity += weights['artists'] * jaccard_artists

        # 3. Tương đồng về DANH MỤC (sử dụng Jaccard)
        categories1 = set(details1.get('category_ids', []))
        categories2 = set(details2.get('category_ids', []))
        if categories1 or categories2:
            intersection_categories = len(categories1.intersection(categories2))
            union_categories = len(categories1.union(categories2))
            jaccard_categories = intersection_categories / union_categories if union_categories > 0 else 0
            total_similarity += weights['categories'] * jaccard_categories
            
        # 4. Tương đồng về GIAI ĐOẠN (sử dụng Jaccard)
        periods1 = set(details1.get('period_ids', []))
        periods2 = set(details2.get('period_ids', []))
        if periods1 or periods2:
            intersection_periods = len(periods1.intersection(periods2))
            union_periods = len(periods1.union(periods2))
            jaccard_periods = intersection_periods / union_periods if union_periods > 0 else 0
            total_similarity += weights['periods'] * jaccard_periods

        # 5. Tương đồng về QUỐC GIA (trùng khớp hoàn toàn)
        nationality1 = details1.get('nationality')
        nationality2 = details2.get('nationality')
        if nationality1 and nationality2 and nationality1 == nationality2:
            total_similarity += weights['nationality'] * 1.0

        # 6. Tương đồng về NGƯỜI ĐÓNG GÓP (trùng khớp hoàn toàn)
        contributor1 = details1.get('contributor_id')
        contributor2 = details2.get('contributor_id')
        if contributor1 and contributor2 and contributor1 == contributor2:
            total_similarity += weights['contributor'] * 1.0

        return total_similarity

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            similarity = self.similarities[i, rating[0]]
            neighbors.append((similarity, rating[1]))

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if simScore > 0:
                simTotal += simScore
                weightedSum += simScore * rating

        if simTotal == 0:
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
