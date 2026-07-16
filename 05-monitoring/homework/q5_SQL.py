"""
Solve question 5 with with SQL.
Query the DB to see which span takes longest.
"""

import sqlite3

con = sqlite3.connect('traces.db')

print(con.execute('''
    SELECT name,
           SUM(end_time - start_time) AS total_ns,
           SUM(end_time - start_time) / 1e9 AS total_sec
    FROM spans
    WHERE name != 'rag'
    GROUP BY name
    ORDER BY total_ns DESC
''').fetchall())