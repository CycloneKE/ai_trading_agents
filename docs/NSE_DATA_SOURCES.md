# NSE price data: where it comes from, and adding a licensed feed

## Where NSE prices come from today

The agent builds one daily price bar per stock per trading day from several
free sources. None of them is licensed, and a change to any website can
leave a gap.

| Source tag | What it is | Code |
|---|---|---|
| `nse_ticker` | The price strip on the NSE's own website | `src/connectors/nse_scraper.py` |
| `nse_pricelist` | The NSE's daily price list (PDF), read and checked | `src/connectors/nse_pricelist.py` |
| `afx_kwayisi`, `afx_history` | Price tables on afx.kwayisi.org | `src/connectors/nse_scraper.py` |
| `aib_market_pulse` | Closing prices in AIB-AXYS's daily Market Pulse | `src/agent/market_pulse.py` |
| `licensed_eod` | A licensed end-of-day feed, once one is bought | `src/connectors/nse_licensed_feed.py` |

## Which source wins a day

When two sources record the same day, the better one is kept, in this order
(`SOURCE_RANK` in `nse_scraper.py`):

1. `licensed_eod` (a licensed feed)
2. `nse_pricelist` (the exchange's own daily list)
3. `nse_ticker` (the exchange's ticker)
4. `nse_website`, then `afx_kwayisi` / `afx_history`, then `aib_market_pulse`

Before, whichever source wrote last won, so a web-page price scraped during
the session could replace the exchange's own end-of-day list. A source may
always update its own bar (the ticker moves during the day), and seed data
never replaces a real price.

## When sources disagree

If two real sources' closes for the same day are more than 3% apart, the
agent logs a warning and lists the day under `source_disagreements` in the
NSE scraper's status (`/api/nse-market`, `scraper`). The better-ranked
source's price is the one kept. A disagreement usually means one site
changed its layout or served a stale page; check the stock on the NSE
website.

## Adding a licensed end-of-day feed

The NSE's Data Services and data vendors sell end-of-day prices, usually as
a daily file. The agent can read such files without any code change.

1. **Buy the feed** (see the checklist below) and get a sample file.
2. **Set the column names** in `config/config.json`, section
   `nse_licensed_feed`. For a file like:

   ```
   Ticker;Trade Date;Open Price;High;Low;Last;Shares
   KCB;13/11/2026;41.00;41.90;40.80;41.40;1,200
   ```

   the settings are:

   ```json
   "nse_licensed_feed": {
     "enabled": true,
     "folder": null,
     "columns": {"symbol": "Ticker", "date": "Trade Date", "open": "Open Price",
                 "high": "High", "low": "Low", "close": "Last", "volume": "Shares"},
     "date_format": "%d/%m/%Y",
     "delimiter": ";"
   }
   ```

   `folder` left as `null` means `data/nse_feed`. Only symbol, date and
   close are required.
3. **Copy each day's file into the folder**, by hand or with the vendor's own
   download tool. The agent imports new or corrected files at the start of
   its next NSE cycle (every 30 minutes in session, every 6 hours outside).
4. **Check the import** in the scraper status (`licensed_feed.last_import`):
   how many bars were read, how many rows could not be read (a sign the
   column names or date format are wrong), and any days where the feed and
   the scrapers disagreed.

The scrapers keep running beside the feed as a cross-check.

Corporate actions (bonus issues, splits) are **not** imported automatically:
they change share counts, so add each one to
`config/nse_corporate_actions.json` after checking the vendor's notice. See
`config/nse_dividends.README.md`.

## What to ask a vendor before buying

A short checklist for the request for quotation:

- **Coverage:** every NSE-listed equity, including suspended and delisted
  stocks, and the start date of the history offered.
- **Fields:** open, high, low, close, volume and turnover per stock per day.
- **Delivery:** file format (CSV preferred), delivery method (SFTP, email or
  download link) and the time each day's file is ready.
- **Corrections:** how a corrected price is re-issued (a new file for the
  same day is imported again automatically).
- **Corporate actions:** whether bonus issues, splits, rights issues and
  dividends (with ex-dates and book closure dates) are included, in what
  format, and how far back. No vendor reviewed for the improvement plan
  documented this in detail, so ask for a sample.
- **Licence terms:** whether use inside an automated trading system and on a
  private dashboard is allowed, and the number of users covered.
- **Price and term:** annual cost, and whether historical data is a
  separate one-off charge.
