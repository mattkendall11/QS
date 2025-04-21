import websocket
import json
import pandas as pd
import numpy as np
import time
import threading
import signal
import sys
from datetime import datetime

# Configuration
symbol = "btcusdt"
depth_url = f"wss://stream.binance.com:9443/ws/{symbol}@depth10@100ms"
data_buffer = []
SAVE_EVERY_N = 10000
RUN_DURATION_SECONDS = 120  # 1 hour
start_time = time.time()


def on_message(ws, message):
    global data_buffer, start_time

    # Check if collection time limit is reached
    now = time.time()
    if now - start_time > RUN_DURATION_SECONDS:
        print("[!] Time's up — closing websocket...")
        ws.close()
        save_data(final=True)
        return

    # Parse the data from the websocket message
    data = json.loads(message)
    timestamp = now
    bids = data['bids']
    asks = data['asks']

    # Extract the relevant features for X and y
    snapshot = {'timestamp': timestamp}

    # Store bid and ask data (X)
    for i in range(10):
        snapshot[f'bid_price_{i}'] = float(bids[i][0])
        snapshot[f'bid_vol_{i}'] = float(bids[i][1])
        snapshot[f'ask_price_{i}'] = float(asks[i][0])
        snapshot[f'ask_vol_{i}'] = float(asks[i][1])

    # Calculate midprice (y)
    snapshot['midprice'] = (snapshot['bid_price_0'] + snapshot['ask_price_0']) / 2

    data_buffer.append(snapshot)

    # Save data periodically
    if len(data_buffer) >= SAVE_EVERY_N:
        save_data()


def prepare_xy_data(df):
    """
    Prepare X (features) and y (target) from the collected data
    """
    # X: All bid/ask prices and volumes
    x_columns = [col for col in df.columns if 'bid_' in col or 'ask_' in col]
    X = df[x_columns]

    # y: Midprice
    y = df['midprice']

    return X, y


def save_data(final=False):
    global data_buffer
    if not data_buffer:
        return

    # Convert collected data to DataFrame
    df = pd.DataFrame(data_buffer)
    print(f"[+] Saving {len(df)} rows")

    # Prepare X and y data
    X, y = prepare_xy_data(df)

    # Generate timestamp for filenames
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = "_final" if final else ""


    # Save complete data (optional)
    full_filename = f"btc_orderbook.csv"
    df.to_csv(full_filename, index=False)


    # Clear buffer
    data_buffer = []


def on_error(ws, error):
    print("Error:", error)


def on_close(ws, *_):
    print("WebSocket closed")


def on_open(ws):
    print("WebSocket opened — collecting order book data...")
    print(f"Will save X features (bid/ask prices and volumes) and y (midprice) every {SAVE_EVERY_N} observations")
    print(f"Collection will run for {RUN_DURATION_SECONDS} seconds")


def exit_gracefully(signum, frame):
    print("\n[!] Terminating early...")
    save_data(final=True)
    sys.exit(0)


if __name__ == "__main__":
    # Setup signal handlers for graceful termination
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    print("[~] Starting Bitcoin order book data collector...")
    print(f"[~] Symbol: {symbol}")

    # Initialize and run websocket
    ws = websocket.WebSocketApp(depth_url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()

