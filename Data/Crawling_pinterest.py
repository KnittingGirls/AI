from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import requests
import os


def pinterest_crawler_basic_login(search_query, download_path, email, password, num_images=1000, min_width=100, min_height=100):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    driver_service = Service("") # 크롬 드라이브 파일 위치
    driver = webdriver.Chrome(service=driver_service, options=chrome_options)

    try:
        driver.get("https://www.pinterest.com/login/")
        time.sleep(5) 

        email_input = driver.find_element(By.ID, "email")
        email_input.send_keys(email)

        password_input = driver.find_element(By.ID, "password")
        password_input.send_keys(password)

        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()
        time.sleep(5) 

        print("로그인 성공")

        base_url = "https://www.pinterest.com/search/pins/?q="
        search_url = base_url + search_query

        driver.get(search_url)
        time.sleep(5) 

        image_urls = set()
        scroll_pause_time = 3  
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0 

        while len(image_urls) < num_images and scroll_attempts < 10:
            images = driver.find_elements(By.TAG_NAME, "img")
            for img in images:
                width = driver.execute_script("return arguments[0].naturalWidth;", img)
                height = driver.execute_script("return arguments[0].naturalHeight;", img)

                if width >= min_width and height >= min_height:  # 최소 크기 조건
                    src = img.get_attribute("src")
                    if src and "pinimg.com" in src: 
                        image_urls.add(src)

            # 스크롤 다운
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:  # 더 이상 스크롤이 진행되지 않을 때
                scroll_attempts += 1
            else:
                scroll_attempts = 0  # 스크롤이 진행되면 초기화
            last_height = new_height

        print(f"크롤링된 이미지 수: {len(image_urls)}")

        os.makedirs(download_path, exist_ok=True)
        for i, url in enumerate(image_urls):
            if i >= num_images:
                break
            try:
                response = requests.get(url, stream=True)
                with open(os.path.join(download_path, f"{i+1}.jpg"), "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"[{i+1}] 다운로드: {url}")
            except Exception as e:
                print(f"[{i+1}] 다운로드 실패 {url}: {e}")

    except Exception as e:
        print(f"크롤링 중 오류 발생: {e}")

    finally:
        driver.quit()


search_query = "sweater knitting patterns"
download_path = "./pinterest_pattern"
email = ""  # Pinterest 이메일
password = ""  # Pinterest 비밀번호
pinterest_crawler_basic_login(search_query, download_path, email, password, min_width=100, min_height=100)



