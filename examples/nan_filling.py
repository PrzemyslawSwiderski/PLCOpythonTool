import pandas as pd

csv_dataFrame = pd.read_csv('sample_data.csv')
print('Dane wejściowe:')
print(csv_dataFrame)
#    Col1  Col2  Col3  Col4  Col5
# 0   1.0   4.7   2.0   1.5   1.0
# 1   NaN  -9.0   0.0   5.2   NaN
# 2  -2.4  -1.0   1.4   8.7   NaN
# 3  -7.1  -3.0   NaN   2.4   NaN
# 4   4.0   8.0   1.4   1.2   6.6
# 5   NaN   3.4   NaN   1.3   NaN
print('Dane po usunięciu wierszy z wartościami NaN:')
print(csv_dataFrame.dropna())
#    Col1  Col2  Col3  Col4  Col5
# 0   1.0   4.7   2.0   1.5   1.0
# 4   4.0   8.0   1.4   1.2   6.6
print('Dane po usunięciu kolumn z wartościami NaN:')
print(csv_dataFrame.dropna(axis=1))
#    Col2  Col4
# 0   4.7   1.5
# 1  -9.0   5.2
# 2  -1.0   8.7
# 3  -3.0   2.4
# 4   8.0   1.2
# 5   3.4   1.3
print('Usunięcie wierszy z mniej niż 3 próbkami:')
print(csv_dataFrame.dropna(thresh=3))
#    Col1  Col2  Col3  Col4  Col5
# 0   1.0   4.7   2.0   1.5   1.0
# 1   NaN  -9.0   0.0   5.2   NaN
# 2  -2.4  -1.0   1.4   8.7   NaN
# 3  -7.1  -3.0   NaN   2.4   NaN
# 4   4.0   8.0   1.4   1.2   6.6
print('Usuwanie wierszy z wartościami NaN w kolumnie Col1:')
print(csv_dataFrame.dropna(subset=['Col1']))
#    Col1  Col2  Col3  Col4  Col5
# 0   1.0   4.7   2.0   1.5   1.0
# 2  -2.4  -1.0   1.4   8.7   NaN
# 3  -7.1  -3.0   NaN   2.4   NaN
# 4   4.0   8.0   1.4   1.2   6.6
print('Uzupełnienie wartości NaN wartością 0:')
print(csv_dataFrame.fillna(0.0))
#    Col1  Col2  Col3  Col4  Col5
# 0   1.0   4.7   2.0   1.5   1.0
# 1   0.0  -9.0   0.0   5.2   0.0
# 2  -2.4  -1.0   1.4   8.7   0.0
# 3  -7.1  -3.0   0.0   2.4   0.0
# 4   4.0   8.0   1.4   1.2   6.6
# 5   0.0   3.4   0.0   1.3   0.0
print('Uzupełnienie wartości NaN średnią z poszczególnych kolumn:')
print(csv_dataFrame.fillna(csv_dataFrame.mean()))
#     Col1  Col2  Col3  Col4  Col5
# 0  1.000   4.7   2.0   1.5   1.0
# 1 -1.125  -9.0   0.0   5.2   3.8
# 2 -2.400  -1.0   1.4   8.7   3.8
# 3 -7.100  -3.0   1.2   2.4   3.8
# 4  4.000   8.0   1.4   1.2   6.6
# 5 -1.125   3.4   1.2   1.3   3.8