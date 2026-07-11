"""
Crypto Day-Trading Signal Bot
==============================
Bot ini TIDAK melakukan auto-trade. Dia hanya membaca data harga dari Binance,
menghitung indikator teknikal, lalu memberi SINYAL beli beserta level
Take Profit (TP) dan Stop Loss (SL). Eksekusi order tetap manual/keputusan kamu.

PENTING:
- Ini bukan jaminan profit. Sinyal berbasis indikator teknikal punya win rate
  terbatas (biasanya 40-60% tergantung kondisi market).
- WAJIB backtest dulu sebelum dipakai uang beneran (lihat fungsi backtest()).
- Jalankan script ini di komputer yang punya akses internet ke api.binance.com.

Cara pakai cepat:
    pip install pandas numpy requests
    python crypto_signal_bot.py --symbol BTCUSDT --interval 15m --mode live
    python crypto_signal_bot.py --symbol BTCUSDT --interval 15m --mode backtest
"""

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# ----------------------------------------------------------------------
# KONFIGURASI TELEGRAM
# Prioritas: baca dari environment variable dulu (aman, dipakai GitHub Actions
# lewat Secrets). Kalau kosong, pakai nilai yang diisi manual di bawah ini
# (buat dijalankan langsung di PC lokal). JANGAN share file ini ke orang lain
# setelah diisi manual (token = password bot).
# ----------------------------------------------------------------------
import os
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ----------------------------------------------------------------------
# 1. AMBIL DATA
# ----------------------------------------------------------------------
def fetch_klines(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    """
    Ambil data candlestick (OHLCV) dari Binance public API.
    interval contoh: '1m', '5m', '15m', '1h', '4h', '1d'
    limit maksimum 1000 per request.
    """
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["open_time", "open", "high", "low", "close", "volume"]]


# ----------------------------------------------------------------------
# 2. INDIKATOR TEKNIKAL
# ----------------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)
    return df


# ----------------------------------------------------------------------
# 3. LOGIKA SINYAL
# ----------------------------------------------------------------------
def generate_signal(df: pd.DataFrame, rr_ratio: float = 2.0) -> dict:
    """
    Cek 2 candle terakhir untuk deteksi golden cross EMA9/EMA21,
    dikonfirmasi RSI di zona sehat (40-70).
    Return dict berisi sinyal + level entry/TP/SL.
    """
    if len(df) < 25:
        return {"signal": "HOLD", "reason": "data belum cukup"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    cross_up = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
    rsi_ok = 40 <= last["rsi14"] <= 70

    if cross_up and rsi_ok:
        entry = last["close"]
        sl = entry - last["atr14"]
        tp = entry + last["atr14"] * rr_ratio
        return {
            "signal": "BUY",
            "time": last["open_time"],
            "entry": round(entry, 4),
            "stop_loss": round(sl, 4),
            "take_profit": round(tp, 4),
            "rsi": round(last["rsi14"], 1),
            "reason": "EMA9 cross up EMA21, RSI netral",
        }

    return {"signal": "HOLD", "rsi": round(last["rsi14"], 1) if not np.isnan(last["rsi14"]) else None}


# ----------------------------------------------------------------------
# 4. BACKTEST SEDERHANA
# ----------------------------------------------------------------------
def backtest(df: pd.DataFrame, rr_ratio: float = 2.0) -> dict:
    """
    Simulasi: setiap kali muncul sinyal BUY, cek candle-candle berikutnya
    apakah harga kena TP dulu atau SL dulu. Hitung win rate & rata2 hasil.
    """
    df = compute_indicators(df)
    trades = []

    for i in range(25, len(df) - 1):
        window = df.iloc[: i + 1]
        sig = generate_signal(window, rr_ratio)
        if sig["signal"] != "BUY":
            continue

        entry, sl, tp = sig["entry"], sig["stop_loss"], sig["take_profit"]
        outcome = None
        for j in range(i + 1, len(df)):
            high, low = df.iloc[j]["high"], df.iloc[j]["low"]
            if low <= sl:
                outcome = "LOSS"
                break
            if high >= tp:
                outcome = "WIN"
                break
        if outcome:
            trades.append(outcome)

    total = len(trades)
    wins = trades.count("WIN")
    win_rate = (wins / total * 100) if total else 0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate_pct": round(win_rate, 1),
    }


# ----------------------------------------------------------------------
# 5. NOTIFIKASI (placeholder - isi token Telegram kalau mau dipakai)
# ----------------------------------------------------------------------
def send_telegram_alert(message: str, bot_token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
    if not bot_token or not chat_id:
        print(f"[ALERT - Telegram belum dikonfigurasi] {message}")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception as e:
        print(f"Gagal kirim Telegram: {e}")


# ----------------------------------------------------------------------
# 6. MODE LIVE (polling berkala)
# ----------------------------------------------------------------------
def run_live(symbol: str, interval: str, poll_seconds: int = 60):
    print(f"Memantau {symbol} ({interval}) setiap {poll_seconds}s... Ctrl+C untuk stop.")
    last_signal_time = None
    while True:
        try:
            df = fetch_klines(symbol, interval, limit=200)
            df = compute_indicators(df)
            sig = generate_signal(df)

            if sig["signal"] == "BUY" and sig.get("time") != last_signal_time:
                last_signal_time = sig["time"]
                msg = (
                    f"[{symbol}] SINYAL BUY @ {sig['entry']}\n"
                    f"TP: {sig['take_profit']} | SL: {sig['stop_loss']}\n"
                    f"RSI: {sig['rsi']} | Alasan: {sig['reason']}"
                )
                print(f"\n{datetime.now()} {msg}\n")
                send_telegram_alert(msg)
            else:
                print(f"{datetime.now()} - {symbol}: HOLD (RSI={sig.get('rsi')})")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(poll_seconds)


def run_once(symbol: str, interval: str):
    """
    Cek sinyal SEKALI SAJA lalu keluar. Cocok dipanggil berkala oleh
    penjadwal eksternal seperti GitHub Actions (cron), bukan loop sendiri.
    """
    df = fetch_klines(symbol, interval, limit=200)
    df = compute_indicators(df)
    sig = generate_signal(df)

    if sig["signal"] == "BUY":
        msg = (
            f"[{symbol}] SINYAL BUY @ {sig['entry']}\n"
            f"TP: {sig['take_profit']} | SL: {sig['stop_loss']}\n"
            f"RSI: {sig['rsi']} | Alasan: {sig['reason']}"
        )
        print(f"{datetime.now()} {msg}")
        send_telegram_alert(msg)
    else:
        print(f"{datetime.now()} - {symbol}: HOLD (RSI={sig.get('rsi')})")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Day-Trading Signal Bot")
    parser.add_argument("--symbol", default="BTCUSDT", help="Pair, contoh: BTCUSDT, ETHUSDT")
    parser.add_argument("--interval", default="15m", help="Timeframe: 1m,5m,15m,1h,4h,1d")
    parser.add_argument("--mode", choices=["live", "backtest", "once"], default="backtest")
    parser.add_argument("--poll", type=int, default=60, help="Interval polling (detik) untuk mode live")
    args = parser.parse_args()

    if args.mode == "once":
        run_once(args.symbol, args.interval)
    elif args.mode == "backtest":
        print(f"Mengambil data historis {args.symbol} ({args.interval})...")
        df = fetch_klines(args.symbol, args.interval, limit=1000)
        result = backtest(df)
        print("\n=== HASIL BACKTEST ===")
        for k, v in result.items():
            print(f"{k}: {v}")
        print("\nCatatan: hasil di masa lalu tidak menjamin hasil sama di masa depan.")
    else:
        run_live(args.symbol, args.interval, args.poll)
