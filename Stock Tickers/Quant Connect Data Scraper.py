import base64
import time
import pandas as pd
from IPython.display import HTML, Javascript

qb = QuantBook()

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    return f"data:text/csv;base64,{payload}"

ticker = 'AMZN'
equity = qb.AddEquity(ticker)
start_time = datetime(2021, 3, 1)
end_time = datetime.now()
history = qb.History(equity.Symbol, start_time, end_time, Resolution.Minute)
if len(history):
    display(Javascript("""
        let link = document.createElement("a");
            link.setAttribute('download', '{}');
            link.href = '{}';
            link.target = '_blank';
            document.body.appendChild(link);
            link.click();
            link.remove();
        """.format(f"{ticker}_2021_March_to_May.csv", create_download_link(history))))
