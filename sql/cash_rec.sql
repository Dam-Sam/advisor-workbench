-- Cash activity reconciliation for income events (dividends and distributions).
WITH j AS (
    SELECT
        COALESCE(b.account_id, c.account_id)       AS account_id,
        COALESCE(b.ticker, c.ticker)               AS ticker,
        COALESCE(b.pay_date, c.pay_date)           AS pay_date,
        COALESCE(b.activity_type, c.activity_type) AS activity_type,
        b.amount AS book_amount,
        c.amount AS cust_amount
    FROM book_cash b
    FULL OUTER JOIN custodian_cash c
      ON b.account_id = c.account_id AND b.ticker = c.ticker
     AND b.pay_date = c.pay_date AND b.activity_type = c.activity_type
)
SELECT
    *,
    CASE
        WHEN book_amount IS NULL THEN 'UNBOOKED_DIVIDEND'
        WHEN cust_amount IS NULL THEN 'DIVIDEND_NOT_AT_CUSTODIAN'
        WHEN abs(book_amount - cust_amount) > $cash_tol THEN 'DIVIDEND_AMOUNT_MISMATCH'
        ELSE 'MATCHED'
    END AS rec_status,
    round(abs(COALESCE(cust_amount, 0) - COALESCE(book_amount, 0)), 2) AS impact_cad
FROM j
ORDER BY rec_status = 'MATCHED', impact_cad DESC;
