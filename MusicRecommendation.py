from dotenv import load_dotenv
load_dotenv()
import psycopg2
from surprise import Dataset, Reader
from collections import defaultdict
import pandas as pd
import os
from psycopg2 import pool
import redis
from urllib.parse import urlparse



# --- CÁC LỚP KẾT NỐI (DatabaseConnection, RedisConnection) KHÔNG THAY ĐỔI ---

# Lớp DatabaseConnection không thay đổi
class DatabaseConnection:
    _instance = None
    _connection_pool = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._connection_pool = pool.SimpleConnectionPool(1, 10, host=os.getenv('DB_HOST'), database=os.getenv('DB_NAME'), user=os.getenv('DB_USERNAME'), password=os.getenv('DB_PASSWORD'), port=os.getenv('DB_PORT'))
        return cls._instance
    def get_connection(self): return self._connection_pool.getconn()
    def release_connection(self, connection): self._connection_pool.putconn(connection)
    def close_all_connections(self): self._connection_pool.closeall()
# --- LỚP MusicRecommendation ĐƯỢC CẬP NHẬT ---

class MusicRecommendation:
    musicID_to_name = {}
    name_to_musicID = {}
    musicID_to_details = {}  # Cấu trúc được mở rộng

    def __init__(self):
        self.db_connection_manager = DatabaseConnection()
        # self.redis_connection = RedisConnection().get_connection()
        # Redis Connection
        REDIS_HOST_NAME = os.getenv("REDIS_HOST_NAME")
        REDIS_PORT = os.getenv("REDIS_PORT")
        REDIS_USERNAME = os.getenv("REDIS_USERNAME")
        REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

        redis_client = redis.StrictRedis(
            host=REDIS_HOST_NAME,
            port=REDIS_PORT,
            username=REDIS_USERNAME,
            password=REDIS_PASSWORD,
            ssl=True,
            decode_responses=True
        )


        self.redis_client = redis_client

    def _connect_db(self):
        return self.db_connection_manager.get_connection()

    def _release_db_connection(self, connection):
        self.db_connection_manager.release_connection(connection)

    def loadMusicData(self):
        """
        Tải dữ liệu điểm nghe nhạc và thông tin chi tiết của bài hát,
        bao gồm nghệ sĩ, danh mục, thể loại và giai đoạn.
        """
        self.musicID_to_name = {}
        self.name_to_musicID = {}
        self.musicID_to_details = {}

        connection = self._connect_db()
        cursor = connection.cursor()

        # Tải dữ liệu điểm nghe nhạc (không thay đổi)
        cursor.execute("SELECT listener_id, music_id, score FROM listener_music_recommend_score")
        ratings = cursor.fetchall()
        reader = Reader(line_format='user item rating', sep=',')
        ratings_df = pd.DataFrame(ratings, columns=['listener_id', 'music_id', 'score'])
        ratings_df.dropna(subset=['score'], inplace=True)
        ratingsDataset = Dataset.load_from_df(ratings_df[['listener_id', 'music_id', 'score']], reader)

        # --- CẬP NHẬT TRUY VẤN SQL ĐỂ JOIN CÁC BẢNG ---
        # Sử dụng LEFT JOIN để đảm bảo tất cả các bài hát đều được lấy ra,
        # ngay cả khi chúng không có thông tin ở một trong các bảng phụ.
        # ARRAY_AGG(DISTINCT ...) để gom các ID liên quan thành một mảng.
        music_details_query = """
        SELECT
            m.id,
            m.name,
            m.nationality,
            m.uploaded_by_id,
            ARRAY_AGG(DISTINCT ma.artist_id) FILTER (WHERE ma.artist_id IS NOT NULL) as artist_ids,
            ARRAY_AGG(DISTINCT mc.category_id) FILTER (WHERE mc.category_id IS NOT NULL) as category_ids,
            ARRAY_AGG(DISTINCT mg.genre_id) FILTER (WHERE mg.genre_id IS NOT NULL) as genre_ids,
            ARRAY_AGG(DISTINCT mp.period_id) FILTER (WHERE mp.period_id IS NOT NULL) as period_ids
        FROM
            mucis m
        LEFT JOIN music_artists ma ON m.id = ma.music_id
        LEFT JOIN music_categories mc ON m.id = mc.music_id
        LEFT JOIN music_genres mg ON m.id = mg.music_id
        LEFT JOIN music_periods mp ON m.id = mp.music_id
        GROUP BY
            m.id, m.name, m.nationality, m.uploaded_by_id
        """
        cursor.execute(music_details_query)
        musics = cursor.fetchall()

        for row in musics:
            musicID, musicName, nationality, contributor_id, artist_ids, category_ids, genre_ids, period_ids = row
            
            self.musicID_to_name[musicID] = musicName
            self.name_to_musicID[musicName] = musicID
            self.musicID_to_details[musicID] = {
                'nationality': nationality,
                'contributor_id': contributor_id,
                # Chuyển đổi giá trị None (nếu có) thành danh sách rỗng
                'artist_ids': artist_ids or [],
                'category_ids': category_ids or [],
                'genre_ids': genre_ids or [],
                'period_ids': period_ids or []
            }

        cursor.close()
        self._release_db_connection(connection)

        return ratingsDataset

    # --- CÁC PHƯƠNG THỨC GETTER CŨ VÀ MỚI ---

    def getMusicName(self, musicID): return self.musicID_to_name.get(musicID, "")
    def getMusicID(self, musicName): return self.name_to_musicID.get(musicName, 0)
    def getNationality(self, musicID): return self.musicID_to_details.get(musicID, {}).get('nationality', "")
    def getContributorID(self, musicID): return self.musicID_to_details.get(musicID, {}).get('contributor_id', "")

    # Getter mới cho các thông tin đã join
    def getArtistIDs(self, musicID):
        """Lấy danh sách ID nghệ sĩ của một bài hát."""
        return self.musicID_to_details.get(musicID, {}).get('artist_ids', [])

    def getCategoryIDs(self, musicID):
        """Lấy danh sách ID danh mục của một bài hát."""
        return self.musicID_to_details.get(musicID, {}).get('category_ids', [])

    def getGenreIDs(self, musicID):
        """Lấy danh sách ID thể loại của một bài hát."""
        return self.musicID_to_details.get(musicID, {}).get('genre_ids', [])

    def getPeriodIDs(self, musicID):
        """Lấy danh sách ID giai đoạn của một bài hát."""
        return self.musicID_to_details.get(musicID, {}).get('period_ids', [])


    # --- CÁC PHƯƠNG THỨC CÒN LẠI KHÔNG THAY ĐỔI ---
    def getListenerRatings(self, listener_id):
        # ... không thay đổi ...
        userRatings = []
        connection = self._connect_db()
        cursor = connection.cursor()
        cursor.execute("SELECT music_id, score FROM listener_music_recommend_score WHERE listener_id = %s", (listener_id,))
        ratings = cursor.fetchall()
        for row in ratings: userRatings.append((row[0], float(row[1])))
        cursor.close()
        self._release_db_connection(connection)
        return userRatings

    def getPopularityRanks(self):
        # ... không thay đổi ...
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        connection = self._connect_db()
        cursor = connection.cursor()
        cursor.execute("SELECT music_id FROM listener_music_recommend_score")
        ratings_data = cursor.fetchall()
        for row in ratings_data: ratings[row[0]] += 1
        rank = 1
        for musicID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[musicID] = rank
            rank += 1
        cursor.close()
        self._release_db_connection(connection)
        return rankings

    def loadListeners(self):
        # ... không thay đổi ...
        connection = self._connect_db()
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT listener_id FROM listener_music_recommend_score")
        users = cursor.fetchall()
        cursor.close()
        self._release_db_connection(connection)
        return [user[0] for user in users]

    def saveRecommendationsToRedis(self, listener_id, music_ids, ttl_seconds=86400):
        # ... không thay đổi ...
        if not self.redis_client: return
        key = f"recommendations:listener:{listener_id}"
        value = ",".join(map(str, music_ids))
        self.redis_client.set(key, value, ex=ttl_seconds)

    def saveAllRecommendationsToRedis(self, all_recommendations, ttl_seconds=86400):
        # # ... không thay đổi ...
        if not self.redis_client: return
        with self.redis_client.pipeline() as pipe:
            for listener_id, music_ids in all_recommendations:
                key = f"sonata_recommendations:listener:{listener_id}"
                value = ",".join(map(str, music_ids))
                pipe.set(key, value, ex=ttl_seconds)
            pipe.execute()

# Ví dụ về cách sử dụng
if __name__ == "__main__":
    music_recommendation = MusicRecommendation()
    
    dataset = music_recommendation.loadMusicData()

    print("Dataset loaded successfully.")

    # Get popularity ranks
    print("Top 5 popular musics:", list(music_recommendation.getPopularityRanks().items())[:5])

    # Get period IDs for a specific music ID
    example_music_id = 6  # Thay thế bằng một music ID hợp lệ
    print(f"Period IDs for music ID {example_music_id}:", music_recommendation.getPeriodIDs(example_music_id))

    # Get artist IDs for a specific music ID
    print(f"Artist IDs for music ID {example_music_id}:", music_recommendation.getArtistIDs(example_music_id))

    # Get category IDs for a specific music ID
    print(f"Category IDs for music ID {example_music_id}:", music_recommendation.getCategoryIDs(example_music_id))

    # Get genre IDs for a specific music ID
    print(f"Genre IDs for music ID {example_music_id}:", music_recommendation.getGenreIDs(example_music_id))

    # Get listener ratings for a specific listener ID
    example_listener_id = 3  # Thay thế bằng một listener ID hợp lệ
    print(f"Ratings for listener ID {example_listener_id}:", music_recommendation.getListenerRatings(example_listener_id))
