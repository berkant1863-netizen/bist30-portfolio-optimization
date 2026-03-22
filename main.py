
# main.py — Quantitative Finance Projesi
# Adım 1 + 2 + 3 — Tam ve çalışır hali

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# SABITLER — Tek yerde tanımla, her yerde kullan
# ══════════════════════════════════════════════════════

HISSELER = [
    "THYAO.IS", "GARAN.IS", "AKBNK.IS", "EREGL.IS", "BIMAS.IS",
    "SISE.IS",  "KCHOL.IS", "SAHOL.IS", "ARCLK.IS", "TUPRS.IS",
]

BASLANGIC    = "2023-01-01"
BITIS        = "2025-01-01"
ISLEM_GUNU   = 252
RF_YILLIK    = 0.40               # Yıllık risksiz faiz
RF_GUNLUK    = RF_YILLIK / ISLEM_GUNU

# ══════════════════════════════════════════════════════
# ADIM 1 — VERİ ALTYAPISI
# ══════════════════════════════════════════════════════

print("=" * 60)
print("ADIM 1 — VERİ ALTYAPISI")
print("=" * 60)

print("Veri çekiliyor...")
ham_veri = yf.download(
    tickers=HISSELER,
    start=BASLANGIC,
    end=BITIS,
    auto_adjust=True,
    progress=False
)

fiyat = ham_veri["Close"].copy()
fiyat = fiyat.ffill().dropna()
print(f"Fiyat verisi: {fiyat.shape[0]} gün x {fiyat.shape[1]} hisse")

log_getiri = np.log(fiyat / fiyat.shift(1)).dropna()
print(f"Log getiri:   {log_getiri.shape[0]} gün x {log_getiri.shape[1]} hisse")

yillik_getiri     = log_getiri.mean() * ISLEM_GUNU
yillik_volatilite = log_getiri.std()  * np.sqrt(ISLEM_GUNU)

print("\nYıllık Getiri ve Volatilite:")
for h in HISSELER:
    print(f"  {h.replace('.IS',''):8s}  "
          f"Getiri: {yillik_getiri[h]*100:6.1f}%   "
          f"Vol: {yillik_volatilite[h]*100:5.1f}%")

# ══════════════════════════════════════════════════════
# ADIM 2 — RİSK METRİKLERİ
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ADIM 2 — RİSK METRİKLERİ")
print("=" * 60)

mu    = log_getiri.mean()
sigma = log_getiri.std()

var_95      = -(mu + norm.ppf(0.05) * sigma) * 100
var_99      = -(mu + norm.ppf(0.01) * sigma) * 100
hist_var_99 = -log_getiri.quantile(0.01) * 100
sharpe_h    = (mu - RF_GUNLUK) / sigma * np.sqrt(ISLEM_GUNU)

def max_drawdown(seri):
    tepe  = seri.cummax()
    dusus = (seri - tepe) / tepe
    return dusus.min() * 100

mdd = fiyat.apply(max_drawdown)

tablo = pd.DataFrame({
    "Getiri %":  (yillik_getiri     * 100).round(1),
    "Vol %":     (yillik_volatilite * 100).round(1),
    "Sharpe":     sharpe_h.round(3),
    "VaR95 %":    var_95.round(3),
    "VaR99 %":    var_99.round(3),
    "HVaR99 %":   hist_var_99.round(3),
    "MDD %":      mdd.round(1),
})

print("\nRİSK METRİKLERİ TABLOSU:")
print(tablo.to_string())

korelasyon = log_getiri.corr()
kisalt     = {h: h.replace(".IS", "") for h in log_getiri.columns}
kor_gorsel = korelasyon.rename(index=kisalt, columns=kisalt)

print("\nKORELASYON MATRİSİ:")
print(kor_gorsel.round(2).to_string())

# Grafik 1
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Adım 2 — Risk Metrikleri", fontsize=13, fontweight='bold')

ax1 = axes[0]
g_pct = yillik_getiri * 100
v_pct = yillik_volatilite * 100
renkler = plt.cm.RdYlGn(
    (g_pct - g_pct.min()) / (g_pct.max() - g_pct.min())
)
ax1.scatter(v_pct, g_pct, c=renkler, s=130, zorder=3)
for h in HISSELER:
    ax1.annotate(h.replace(".IS", ""), (v_pct[h], g_pct[h]),
                 fontsize=8, xytext=(4, 4), textcoords='offset points')
ax1.axhline(y=40, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.text(v_pct.min(), 41.5, "Rf = %40", fontsize=8, color='gray')
ax1.set_xlabel("Yıllık Volatilite (%)")
ax1.set_ylabel("Yıllık Getiri (%)")
ax1.set_title("Risk / Getiri Haritası")
ax1.grid(alpha=0.2)

ax2 = axes[1]
sns.heatmap(kor_gorsel, ax=ax2, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            linewidths=0.3, annot_kws={"size": 7})
ax2.set_title("Korelasyon Matrisi")

plt.tight_layout()
plt.savefig("risk_metrikleri.png", dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════
# ADIM 3 — PORTFÖY OPTİMİZASYONU
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ADIM 3 — PORTFÖY OPTİMİZASYONU")
print("=" * 60)

N          = len(HISSELER)
kov_matris = log_getiri.cov() * ISLEM_GUNU

# ── Portföy metrik fonksiyonları ─────────────────────

def portfoy_getiri(w):
    return np.dot(w, yillik_getiri)

def portfoy_volatilite(w):
    return np.sqrt(np.dot(w.T, np.dot(kov_matris, w)))

def portfoy_sharpe(w):
    return (portfoy_getiri(w) - RF_YILLIK) / portfoy_volatilite(w)

def negatif_sharpe(w):
    return -portfoy_sharpe(w)

# ── Kısıt ve sınırlar ────────────────────────────────

kisitlar = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
sinirlar = tuple((0.0, 0.40) for _ in range(N))
w0       = np.array([1/N] * N)

# ── 1. Eşit ağırlıklı portföy ────────────────────────

print("\n--- Benchmark: Eşit Ağırlıklı Portföy ---")
g_esit = portfoy_getiri(w0)
v_esit = portfoy_volatilite(w0)
s_esit = portfoy_sharpe(w0)
print(f"Getiri:     %{g_esit*100:.1f}")
print(f"Volatilite: %{v_esit*100:.1f}")
print(f"Sharpe:     {s_esit:.3f}")

# ── 2. Maksimum Sharpe ───────────────────────────────

print("\n--- Optimizasyon: Maksimum Sharpe ---")
sonuc   = minimize(negatif_sharpe, w0, method="SLSQP",
                   bounds=sinirlar, constraints=kisitlar,
                   options={"maxiter": 1000})
w_sharpe = sonuc.x
g_opt    = portfoy_getiri(w_sharpe)
v_opt    = portfoy_volatilite(w_sharpe)
s_opt    = portfoy_sharpe(w_sharpe)
print(f"Getiri:     %{g_opt*100:.1f}")
print(f"Volatilite: %{v_opt*100:.1f}")
print(f"Sharpe:     {s_opt:.3f}")

print("\nOptimal Ağırlıklar:")
for h, w in zip(HISSELER, w_sharpe):
    bar = "█" * int(w * 50)
    print(f"  {h.replace('.IS',''):8s}  %{w*100:5.1f}  {bar}")

print("\n--- Detay: Sıfır olmayan ağırlıklar ---")
for h, w in zip(HISSELER, w_sharpe):
    if w > 0.001:
        print(f"  {h.replace('.IS',''):8s}  %{w*100:.1f}")



# ── 3. Minimum Varyans ───────────────────────────────

print("\n--- Optimizasyon: Minimum Varyans ---")
sonuc_mv = minimize(lambda w: portfoy_volatilite(w), w0,
                    method="SLSQP", bounds=sinirlar,
                    constraints=kisitlar,
                    options={"maxiter": 1000})
w_mv  = sonuc_mv.x
g_mv  = portfoy_getiri(w_mv)
v_mv  = portfoy_volatilite(w_mv)
s_mv  = portfoy_sharpe(w_mv)
print(f"Getiri:     %{g_mv*100:.1f}")
print(f"Volatilite: %{v_mv*100:.1f}")
print(f"Sharpe:     {s_mv:.3f}")

# ── 4. Karşılaştırma tablosu ─────────────────────────

print("\n" + "=" * 60)
print("PORTFÖY KARŞILAŞTIRMASI")
print("=" * 60)
print(f"{'Portföy':<20} {'Getiri':>10} {'Volatilite':>12} {'Sharpe':>10}")
print("-" * 55)
print(f"{'Eşit Ağırlık':<20} "
      f"{g_esit*100:>9.1f}%  {v_esit*100:>10.1f}%  {s_esit:>10.3f}")
print(f"{'Min Varyans':<20} "
      f"{g_mv*100:>9.1f}%  {v_mv*100:>10.1f}%  {s_mv:>10.3f}")
print(f"{'Maks Sharpe':<20} "
      f"{g_opt*100:>9.1f}%  {v_opt*100:>10.1f}%  {s_opt:>10.3f}")

# ── 5. Efficient Frontier ────────────────────────────

print("\nEfficient Frontier hesaplanıyor...")
hedef_getiriler = np.linspace(yillik_getiri.min(),
                               yillik_getiri.max(), 50)
frontier_vol, frontier_get = [], []

for hedef in hedef_getiriler:
    k = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, h=hedef: portfoy_getiri(w) - h}
    ]
    r = minimize(lambda w: portfoy_volatilite(w), w0,
                 method="SLSQP", bounds=sinirlar,
                 constraints=k, options={"maxiter": 500})
    if r.success:
        frontier_vol.append(portfoy_volatilite(r.x) * 100)
        frontier_get.append(portfoy_getiri(r.x) * 100)

# ── 6. Efficient Frontier Grafiği ────────────────────

fig2, ax = plt.subplots(figsize=(10, 7))

# 3000 rastgele portföy
np.random.seed(42)
r_get, r_vol, r_shr = [], [], []
for _ in range(3000):
    w_r = np.random.dirichlet(np.ones(N))
    r_get.append(portfoy_getiri(w_r) * 100)
    r_vol.append(portfoy_volatilite(w_r) * 100)
    r_shr.append(portfoy_sharpe(w_r))

sc = ax.scatter(r_vol, r_get, c=r_shr, cmap='RdYlGn',
                alpha=0.3, s=8, zorder=1)
plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

ax.plot(frontier_vol, frontier_get, 'b-',
        linewidth=2, label='Efficient Frontier', zorder=2)

ax.scatter(v_esit*100, g_esit*100, s=200, color='gray',
           marker='D', zorder=5,
           label=f'Eşit Ağırlık  (Sharpe: {s_esit:.2f})')
ax.scatter(v_mv*100,   g_mv*100,   s=200, color='blue',
           marker='s', zorder=5,
           label=f'Min Varyans  (Sharpe: {s_mv:.2f})')
ax.scatter(v_opt*100,  g_opt*100,  s=250, color='green',
           marker='*', zorder=5,
           label=f'Maks Sharpe  (Sharpe: {s_opt:.2f})')

ax.axhline(y=RF_YILLIK*100, color='red', linestyle='--',
           alpha=0.5, linewidth=1, label='Risksiz Faiz %40')
ax.set_xlabel("Yıllık Volatilite (%)", fontsize=12)
ax.set_ylabel("Yıllık Getiri (%)",     fontsize=12)
ax.set_title("Efficient Frontier — BIST-30 Portföy Optimizasyonu",
             fontsize=13)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150, bbox_inches='tight')
plt.show()
print("Grafik kaydedildi: efficient_frontier.png")
print("\n" + "=" * 60)
print("MAKSİMUM SHARPE — OPTİMAL AĞIRLIKLAR")
print("=" * 60)
for h, w in zip(HISSELER, w_sharpe):
    bar = "█" * int(w * 50)
    print(f"  {h.replace('.IS',''):8s}  %{w*100:5.1f}  {bar}")
print("\nAdım 1 + 2 + 3 tamamlandı.")

# ══════════════════════════════════════════════════════
# ADIM 4 — BACKTESTING (Walk-Forward Test)
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ADIM 4 — BACKTESTING")
print("=" * 60)

# ── Veriyi böl ───────────────────────────────────────
# %70 eğitim (optimize et), %30 test (gerçek performans)
bolme = int(len(log_getiri) * 0.70)

egitim_getiri = log_getiri.iloc[:bolme]
test_getiri   = log_getiri.iloc[bolme:]
test_fiyat    = fiyat.iloc[bolme:]

print(f"Eğitim seti: {egitim_getiri.shape[0]} gün "
      f"({egitim_getiri.index[0].date()} - "
      f"{egitim_getiri.index[-1].date()})")
print(f"Test seti:   {test_getiri.shape[0]} gün "
      f"({test_getiri.index[0].date()} - "
      f"{test_getiri.index[-1].date()})")

# ── Eğitim verisinde optimize et ─────────────────────
kov_egitim      = egitim_getiri.cov() * ISLEM_GUNU
yillik_g_egitim = egitim_getiri.mean() * ISLEM_GUNU

def portfoy_sharpe_egitim(w):
    g = np.dot(w, yillik_g_egitim)
    v = np.sqrt(np.dot(w.T, np.dot(kov_egitim, w)))
    return (g - RF_YILLIK) / v

sonuc_bt = minimize(
    lambda w: -portfoy_sharpe_egitim(w),
    w0, method="SLSQP",
    bounds=sinirlar,
    constraints=kisitlar,
    options={"maxiter": 1000}
)
w_bt = sonuc_bt.x

print("\nEğitim verisinde bulunan optimal ağırlıklar:")
for h, w in zip(HISSELER, w_bt):
    if w > 0.001:
        print(f"  {h.replace('.IS',''):8s}  %{w*100:.1f}")

# ── Test döneminde performansı ölç ────────────────────

# Günlük portföy getirisi = ağırlıkların log getiriyle çarpımı
test_portfoy_getiri = test_getiri.dot(w_bt)
test_esit_getiri    = test_getiri.dot(w0)

# Kümülatif getiri = her günün üzerine birikmesi
kumulatif_opt  = (1 + test_portfoy_getiri).cumprod()
kumulatif_esit = (1 + test_esit_getiri).cumprod()

# Test dönemi metrikleri
def donem_metrikleri(gunluk_getiri, isim):
    toplam    = (1 + gunluk_getiri).prod() - 1
    yillik_g  = gunluk_getiri.mean()  * ISLEM_GUNU
    yillik_v  = gunluk_getiri.std()   * np.sqrt(ISLEM_GUNU)
    sharpe    = (yillik_g - RF_YILLIK) / yillik_v

    # Max Drawdown
    kum = (1 + gunluk_getiri).cumprod()
    tepe = kum.cummax()
    mdd  = ((kum - tepe) / tepe).min() * 100

    print(f"\n{isim}:")
    print(f"  Toplam Getiri:  %{toplam*100:.1f}")
    print(f"  Yıllık Getiri:  %{yillik_g*100:.1f}")
    print(f"  Yıllık Vol:     %{yillik_v*100:.1f}")
    print(f"  Sharpe:         {sharpe:.3f}")
    print(f"  Max Drawdown:   %{mdd:.1f}")
    return toplam, yillik_g, yillik_v, sharpe, mdd

print("\n" + "=" * 60)
print("TEST DÖNEMİ SONUÇLARI (Out-of-Sample)")
print("=" * 60)

met_opt  = donem_metrikleri(test_portfoy_getiri, "Maks Sharpe Portföyü")
met_esit = donem_metrikleri(test_esit_getiri,    "Eşit Ağırlıklı Portföy")

print(f"\nSharpe İyileşmesi: "
      f"{met_opt[3]:.3f} vs {met_esit[3]:.3f} "
      f"({((met_opt[3]-met_esit[3])/abs(met_esit[3])*100):.0f}% daha iyi)")

# ── Grafik: Kümülatif Getiri ─────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 5))

ax3.plot(kumulatif_opt.index,  kumulatif_opt,
         label=f"Maks Sharpe (Sharpe: {met_opt[3]:.2f})",
         color='green', linewidth=2)
ax3.plot(kumulatif_esit.index, kumulatif_esit,
         label=f"Eşit Ağırlık (Sharpe: {met_esit[3]:.2f})",
         color='gray',  linewidth=1.5, linestyle='--')

ax3.axhline(y=1, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax3.set_xlabel("Tarih")
ax3.set_ylabel("Kümülatif Getiri (1 = başlangıç)")
ax3.set_title("Adım 4 — Backtesting: Out-of-Sample Performans")
ax3.legend(fontsize=10)
ax3.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("backtesting.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik kaydedildi: backtesting.png")
print("\nAdım 4 tamamlandı.")

# ══════════════════════════════════════════════════════
# ADIM 5 — BİNANCE API: GERÇEK ZAMANLI VERİ
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ADIM 5 — BİNANCE API")
print("=" * 60)

from binance.client import Client
from datetime import datetime, timedelta
import time

# API key gerekmez — public veri kullanıyoruz
client = Client("", "")

# ── 1. Anlık fiyatları çek ────────────────────────────
print("\nAnlık fiyatlar çekiliyor...")

semboller = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

print("\n--- Binance Spot Fiyatları ---")
for sembol in semboller:
    ticker = client.get_symbol_ticker(symbol=sembol)
    fiyat_usdt = float(ticker["price"])
    print(f"  {sembol:10s}  ${fiyat_usdt:,.2f}")

# ── 2. BTC'nin son 30 günlük verisi ───────────────────
print("\nBTC son 30 günlük veri çekiliyor...")

btc_mumlar = client.get_historical_klines(
    symbol    = "BTCUSDT",
    interval  = Client.KLINE_INTERVAL_1DAY,
    start_str = "30 days ago UTC"
)

# Mum verisi: [açılış zamanı, açılış, yüksek, düşük, kapanış, hacim, ...]
btc_kapanis = [float(mum[4]) for mum in btc_mumlar]
btc_tarih   = [datetime.fromtimestamp(mum[0]/1000) for mum in btc_mumlar]

btc_df = pd.DataFrame({
    "tarih":   btc_tarih,
    "kapanis": btc_kapanis
}).set_index("tarih")

# BTC log getirileri
btc_log_getiri = np.log(btc_df["kapanis"] / btc_df["kapanis"].shift(1)).dropna()

# ── 3. BTC için risk metrikleri ───────────────────────
btc_yillik_g   = btc_log_getiri.mean() * ISLEM_GUNU * 100
btc_yillik_vol = btc_log_getiri.std()  * np.sqrt(ISLEM_GUNU) * 100
btc_sharpe     = (btc_yillik_g/100 - RF_YILLIK) / (btc_yillik_vol/100)

btc_var_99 = -(btc_log_getiri.mean() +
               norm.ppf(0.01) * btc_log_getiri.std()) * 100

print("\n--- BTC Risk Metrikleri (Son 30 Gün) ---")
print(f"  Yıllık Getiri:  %{btc_yillik_g:.1f}")
print(f"  Yıllık Vol:     %{btc_yillik_vol:.1f}")
print(f"  Sharpe:         {btc_sharpe:.3f}")
print(f"  VaR %99 günlük: %{btc_var_99:.2f}")

# ── 4. ETH'nin son 30 günlük verisi ───────────────────
eth_mumlar = client.get_historical_klines(
    symbol    = "ETHUSDT",
    interval  = Client.KLINE_INTERVAL_1DAY,
    start_str = "30 days ago UTC"
)

eth_kapanis    = [float(mum[4]) for mum in eth_mumlar]
eth_log_getiri = pd.Series(eth_kapanis).pct_change().dropna()
eth_log_getiri = np.log(
    pd.Series(eth_kapanis) / pd.Series(eth_kapanis).shift(1)
).dropna()

eth_yillik_g   = eth_log_getiri.mean() * ISLEM_GUNU * 100
eth_yillik_vol = eth_log_getiri.std()  * np.sqrt(ISLEM_GUNU) * 100
eth_sharpe     = (eth_yillik_g/100 - RF_YILLIK) / (eth_yillik_vol/100)

print("\n--- ETH Risk Metrikleri (Son 30 Gün) ---")
print(f"  Yıllık Getiri:  %{eth_yillik_g:.1f}")
print(f"  Yıllık Vol:     %{eth_yillik_vol:.1f}")
print(f"  Sharpe:         {eth_sharpe:.3f}")

# ── 5. Karşılaştırma: BIST Portföyü vs Kripto ─────────
print("\n" + "=" * 60)
print("KARŞILAŞTIRMA: BIST OPTİMİZE PORTFÖY vs KRİPTO")
print("=" * 60)
print(f"{'Varlık':<20} {'Yıllık Getiri':>15} {'Volatilite':>12} {'Sharpe':>10}")
print("-" * 60)
print(f"{'Maks Sharpe (BIST)':<20} "
      f"{g_opt*100:>14.1f}%  "
      f"{v_opt*100:>11.1f}%  "
      f"{s_opt:>10.3f}")
print(f"{'BTC':<20} "
      f"{btc_yillik_g:>14.1f}%  "
      f"{btc_yillik_vol:>11.1f}%  "
      f"{btc_sharpe:>10.3f}")
print(f"{'ETH':<20} "
      f"{eth_yillik_g:>14.1f}%  "
      f"{eth_yillik_vol:>11.1f}%  "
      f"{eth_sharpe:>10.3f}")

# ── 6. Anlık VaR Monitörü ─────────────────────────────
print("\n" + "=" * 60)
print("CANLI VaR MONİTÖRÜ (5 güncelleme)")
print("=" * 60)
print("Her 3 saniyede bir BTC fiyatı çekilip VaR hesaplanıyor...\n")

PORTFOY_DEGERI = 100_000   # 100.000 TL varsayımsal portföy

for i in range(5):
    # Anlık fiyat
    ticker    = client.get_symbol_ticker(symbol="BTCUSDT")
    btc_fiyat = float(ticker["price"])

    # 24 saatlik istatistik
    stats_24h = client.get_ticker(symbol="BTCUSDT")
    degisim   = float(stats_24h["priceChangePercent"])

    # Portföy VaR (BIST optimize portföyü için)
    # Son volatilite bilgisi kullanılıyor
    portfoy_sigma = v_opt / np.sqrt(ISLEM_GUNU)   # Günlük
    var_tl = PORTFOY_DEGERI * portfoy_sigma * norm.ppf(0.99)

    zaman = datetime.now().strftime("%H:%M:%S")
    print(f"[{zaman}]  BTC: ${btc_fiyat:>10,.0f}  "
          f"24h: {degisim:>+6.2f}%  |  "
          f"Portföy VaR(99%): {var_tl:>8,.0f} TL")

    if i < 4:
        time.sleep(3)

print("\nAdım 5 tamamlandı.")
print("\n" + "=" * 60)
print("TÜM ADIMLAR TAMAMLANDI")
print("=" * 60)
print("Proje dosyaları:")
print("  main.py               — Ana kod")
print("  risk_metrikleri.png   — Risk/Getiri haritası")
print("  efficient_frontier.png— Portföy optimizasyonu")
print("  backtesting.png       — Out-of-sample test")


