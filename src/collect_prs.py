from dotenv import load_dotenv
import os
import requests
import time
import pandas as pd
url="https://api.github.com/repos/pandas-dev/pandas/pulls"
load_dotenv()

token = os.getenv("GITHUB_TOKEN")

headers={"accept":"application/vnd.github+json","Authorization": f"Bearer {token}"}
page=1
records=[]
params={'page':page,'per_page':100,'state':'all'}
with requests.Session() as session:
    while True:
        try:
            response=session.get(url,headers=headers,params=params)
            response.raise_for_status()
            data=response.json()
            if not data:
                break
            records.extend(data)
            print(f"Fetched page {page}...")
            page=page+1
            params["page"]=page
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            print(f"Error:{e}")
            break



print(f"\nFinished! Total PRs fetched: {len(records)}")
df=pd.json_normalize(records)
print(df.head())
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/pandas_prs.csv", index=False)
print(page)
print(response.url)