SELECT CONCAT(mid, '/', speed) mid,
       CONCAT(uid, '/', year)  uid,
       accuracy
FROM opal_active_scores;
