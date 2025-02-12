import time
import requests

GITHUB_RAW_URLS = [
    "https://raw.githubusercontent.com/DaiRudy/RUDYComEconHW1/main/HW1.3.2.py",
    "https://raw.githubusercontent.com/DaiRudy/RUDYComEconHW1/main/HW1.4.1.py",
    "https://raw.githubusercontent.com/DaiRudy/RUDYComEconHW1/main/HW1.4.2.py"
]

def fetch_and_run(script_url):
    """
    Fetch a Python script from a GitHub raw URL and execute it.
    Returns the execution time in seconds.
    """
    start = time.time()
    
    response = requests.get(script_url)
    if response.status_code == 200:
        script_content = response.text

        # 过滤掉可能的图片弹窗（如matplotlib）
        script_content = script_content.replace("plt.show()", "")  
        
        # 执行代码
        exec(script_content, {"__name__": "__main__"})
    
    else:
        print(f"Failed to fetch {script_url}, Status Code: {response.status_code}")
        return None

    end = time.time()
    return end - start

if __name__ == "__main__":
    for url in GITHUB_RAW_URLS:
        elapsed_time = fetch_and_run(url)
        if elapsed_time is not None:
            print(f"Executed {url}\nTime used: {elapsed_time:.4f} seconds\n")

