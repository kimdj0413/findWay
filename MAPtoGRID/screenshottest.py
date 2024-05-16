from selenium import webdriver
import time

driver = webdriver.Chrome()
driver.get("https://map.naver.com/p?c=15.00,0,0,1,dh")
time.sleep(3)
width = driver.execute_script("return document.body.scrollWidth") #스크롤 할 수 있는 최대 넓이
height = driver.execute_script("return document.body.scrollHeight") #스크롤 할 수 있는 최대 높이
driver.set_window_size(width, height) #스크롤 할 수 있는 모든 부분을 지정
driver.save_screenshot("testmap.png")