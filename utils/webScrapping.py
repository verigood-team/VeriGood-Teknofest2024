import requests
from bs4 import BeautifulSoup
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
base_urls = [
    "https://www.sikayetvar.com/turkcell",
    "https://www.sikayetvar.com/superonline",
]

for base_url in base_urls:
    
    all_links = []

    for page_num in range(1, 26):
        url = f"{base_url}?page={page_num}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            h2_elements = soup.select('h2.complaint-title')

            for h2 in h2_elements:
                a_elements = h2.find_all('a')
                for a in a_elements:
                    all_links.append(a['href'])

            print(f"Sayfa {page_num} alındı.")
        else:
            print(f"Hata: {response.status_code} - Sayfa alınamadı: {url}")

        time.sleep(1)  

    for link in all_links:
        full_url = f"https://www.sikayetvar.com{link}"
        response = requests.get(full_url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            description_div = soup.find('div', class_='complaint-detail-description')

            if description_div:
                comments = description_div.find_all('p')

                comment_texts = [comment.text for comment in comments]

                comment_texts = ' '.join(comment_texts)
                comment_texts = ' '.join(comment_texts.split())

                with open(f"{base_url.split('/')[-1]}.txt", 'a', encoding='utf-8') as file:
                    file.write(comment_texts + '\n')

            else:
                print(f"Yorumlar alınamadı: {full_url}")
        else:
            print(f"Hata: {response.status_code} - Sayfa alınamadı: {full_url}")

        time.sleep(1) 
