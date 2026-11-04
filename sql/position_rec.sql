-- Position reconciliation: book positions (opening + settled book trades) vs the custodian position file.
-- Each position break is then checked against trade breaks on the same account and security:
-- a break with a trade-level cause is "explained"; one without needs investigation.
WITH book_moves AS (
    SELECT account_id, ticker, quantity FROM opening_positions
    UNION ALL
    SELECT account_id, ticker, CASE side WHEN 'BUY' THEN quantity ELSE -quantity END
    FROM book_trades
    WHERE settle_date <= $as_of
),
book AS (
    SELECT account_id, ticker, round(sum(quantity), 3) AS book_qty
    FROM book_moves GROUP BY account_id, ticker
),
pos AS (
    SELECT
        COALESCE(b.account_id, c.account_id) AS account_id,
        COALESCE(b.ticker, c.ticker)         AS ticker,
        COALESCE(b.book_qty, 0)              AS book_qty,
        COALESCE(c.quantity, 0)              AS cust_qty
    FROM book b
    FULL OUTER JOIN custodian_positions c USING (account_id, ticker)
),
cause AS (
    SELECT account_id, ticker, string_agg(rec_status || ' ' || trade_ref, ', ' ORDER BY trade_ref) AS explained_by
    FROM trade_rec
    WHERE rec_status NOT IN ('MATCHED', 'PRICE_MISMATCH')
    GROUP BY account_id, ticker
)
SELECT
    p.account_id, p.ticker, p.book_qty, p.cust_qty,
    round(p.cust_qty - p.book_qty, 3) AS qty_diff,
    k.explained_by,
    k.explained_by IS NOT NULL AS explained
FROM pos p
LEFT JOIN cause k USING (account_id, ticker)
WHERE abs(p.cust_qty - p.book_qty) > $qty_tol
ORDER BY explained, p.account_id, p.ticker;
