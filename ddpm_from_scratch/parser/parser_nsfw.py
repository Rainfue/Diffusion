import requests
import os
import time
import json
from tqdm import tqdm

class DanbooruParser:
    def __init__(self, tags: str, output_dir: str = "downloaded", 
                 limit_per_page: int = 100, max_pages: int = 5, 
                 min_score: int = 20, rating: str = "g", 
                 delay: float = 1.0, start_page: int = 1):  # Добавлен параметр start_page
        """
        :param start_page: Начальная страница для парсинга
        """
        self.tags = tags
        self.output_dir = output_dir
        self.limit = min(limit_per_page, 200)
        self.max_pages = max_pages
        self.min_score = min_score
        self.rating = rating
        self.delay = delay
        self.start_page = start_page  # Сохраняем начальную страницу
        self.base_url = "https://danbooru.donmai.us/posts.json"
        self.metadata = []
        os.makedirs(output_dir, exist_ok=True)

    def fetch_page(self, page: int):
        """Загрузка страницы с постами"""
        params = {
            "tags": self.tags,
            "page": page,
            "limit": self.limit,
            "rating": self.rating
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ошибка при загрузке страницы {page}: {str(e)}")
            return None

    def download_image(self, url: str, filename: str):
        """Скачивание и сохранение изображения"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"Ошибка загрузки {url}: {str(e)}")
            return False

    def save_metadata(self):
        """Сохранение метаданных в JSON"""
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

    def run(self):
        """Основной цикл парсинга"""
        print(f"🚀 Начало парсинга тега: {self.tags} со страницы {self.start_page}")
        downloaded_ids = set()
        total_downloaded = 0

        # Загружаем существующие метаданные (если есть)
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
                downloaded_ids = {item["id"] for item in existing_metadata}
            print(f"ℹ️ Обнаружено {len(downloaded_ids)} уже скачанных изображений")

        for page in range(self.start_page, self.start_page + self.max_pages):
            print(f"📖 Страница {page}/{self.start_page + self.max_pages - 1}")
            data = self.fetch_page(page)
            
            if not data:  # Если страница пустая или ошибка
                print(f"ℹ️ Достигнут конец результатов на странице {page}")
                break

            page_downloaded = 0
            for post in tqdm(data, desc="Скачивание"):
                # Проверка условий
                if post["id"] in downloaded_ids:
                    continue
                if post["score"] < self.min_score:
                    continue
                if not post.get("file_url"):
                    continue

                # Подготовка данных
                file_ext = os.path.splitext(post["file_url"])[1]
                filename = f"{post['id']}{file_ext}"
                filepath = os.path.join(self.output_dir, filename)

                # Скачивание
                if self.download_image(post["file_url"], filepath):
                    downloaded_ids.add(post["id"])
                    total_downloaded += 1
                    page_downloaded += 1
                    
                    # Сохранение метаданных
                    metadata_entry = {
                        "id": post["id"],
                        "tags": post["tag_string"],
                        "score": post["score"],
                        "rating": post["rating"],
                        "source": post.get("source", ""),
                        "file_path": filename,
                        "page": page  # Добавляем номер страницы в метаданные
                    }
                    self.metadata.append(metadata_entry)

            print(f"ℹ️ На странице {page} скачано: {page_downloaded} изображений")
            
            if page_downloaded == 0:
                print("ℹ️ Нет новых изображений для скачивания")
                break
                
            time.sleep(self.delay)  # Защита от бана

        # Объединяем новые данные с существующими
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
            self.metadata = existing_metadata + self.metadata
        
        self.save_metadata()
        print(f"✅ Готово! Всего скачано изображений: {total_downloaded}")
        print(f"💾 Метаданные сохранены в {metadata_path}")

if __name__ == "__main__":
    # Настройки парсера
    parser = DanbooruParser(
        tags="ayanami_rei", 
        output_dir="rei_images",
        limit_per_page=50,
        max_pages=300,
        min_score=5,
        rating="g",
        delay=3,
        start_page=40  # Начинаем со второй страницы
    )
    parser.run()