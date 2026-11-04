-- Trade-level reconciliation: internal book of record vs custodian confirmations.
-- Only trades that reached confirmation are compared; executed/allocated trades are still in flight.
-- Matching key: our trade_id travels to the custodian as client_ref.
WITH book AS (
    SELECT * FROM book_trades WHERE status IN ('CONFIRMED', 'SETTLED')
),
matched AS (
    SELECT
        COALESCE(b.trade_id, c.client_ref, c.cust_trade_id) AS trade_ref,
        c.cust_trade_id,
        COALESCE(b.account_id, c.account_id)                AS account_id,
        COALESCE(b.ticker, c.ticker)                        AS ticker,
        COALESCE(b.trade_date, c.trade_date)                AS trade_date,
        b.side        AS book_side,        c.side        AS cust_side,
        b.quantity    AS book_qty,         c.quantity    AS cust_qty,
        b.price       AS book_price,       c.price       AS cust_price,
        b.settle_date AS book_settle_date, c.settle_date AS cust_settle_date,
        b.trade_id IS NOT NULL  AS in_book,
        c.cust_trade_id IS NOT NULL AS at_custodian
    FROM book b
    FULL OUTER JOIN custodian_trades c ON b.trade_id = c.client_ref
)
SELECT
    *,
    CASE
        WHEN NOT at_custodian                                   THEN 'MISSING_AT_CUSTODIAN'
        WHEN NOT in_book                                        THEN 'MISSING_IN_BOOK'
        WHEN book_side <> cust_side
          OR abs(book_qty - cust_qty) > $qty_tol                THEN 'QTY_MISMATCH'
        WHEN abs(book_price - cust_price)
             > greatest($price_abs_tol, book_price * $price_rel_tol) THEN 'PRICE_MISMATCH'
        WHEN book_settle_date <> cust_settle_date               THEN 'SETTLE_DATE_MISMATCH'
        ELSE 'MATCHED'
    END AS rec_status,
    -- Cash or exposure at stake, in CAD
    round(CASE
        WHEN NOT at_custodian THEN book_qty * book_price
        WHEN NOT in_book      THEN cust_qty * cust_price
        WHEN book_side <> cust_side OR abs(book_qty - cust_qty) > $qty_tol
                              THEN abs(book_qty - cust_qty) * book_price
        WHEN abs(book_price - cust_price) > greatest($price_abs_tol, book_price * $price_rel_tol)
                              THEN abs(book_price - cust_price) * book_qty
        WHEN book_settle_date <> cust_settle_date THEN book_qty * book_price
        ELSE 0
    END, 2) AS impact_cad
FROM matched
ORDER BY rec_status = 'MATCHED', impact_cad DESC;
