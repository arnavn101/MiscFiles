import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


infoLink = 'https://ballotpedia.org/Net_worth_of_United_States_Senators_and_Representatives'

tableSenators = pd.read_html(infoLink, match='Estimated Net Worth of U.S. Senators', thousands=',')[0]
tableSenators.columns = [x[1] for x in tableSenators.columns]

tableRepr = pd.read_html(infoLink, match='Estimated Net Worth of U.S. Representatives', thousands=',')[0]
tableRepr.columns = [x[1] for x in tableRepr.columns]

listCols = [col for col in tableSenators.columns if 'AvgValue' in col]
for col in listCols:
    tableSenators[col] = pd.to_numeric(tableSenators[col].replace({'\$': '', ',': ''}, regex=True),
                                       errors='coerce').astype('float')
    tableRepr[col] = pd.to_numeric(tableRepr[col].replace({'\$': '', ',': ''}, regex=True),
                                   errors='coerce').astype('float')

listNetWorthSenators = tableSenators.mean(axis=1).to_numpy()
listNetWorthRepr = tableRepr.mean(axis=1).to_numpy()

listNetWorthSenators = reject_outliers(listNetWorthSenators)
listNetWorthRepr = reject_outliers(listNetWorthRepr)

lenSenators = len(listNetWorthSenators)
lenRepr = len(listNetWorthRepr)

listNetWorthSenators = listNetWorthSenators[np.random.choice(lenSenators, size=100, replace=False)]
listNetWorthRepr = listNetWorthRepr[np.random.choice(lenRepr, size=100, replace=False)]

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(hspace=1.1, wspace=1.1)
fig.suptitle('Data for both Senate and House of Representatives', color='green')

# First Plot
ax1.title.set_text('Members of the Senate')
ax1.title.set_color('red')
ax1.set_xlabel('Income range - $10^6', color='purple')
ax1.set_ylabel('Frequency', color='purple')
countsS, binsS, barsS = ax1.hist(listNetWorthSenators, alpha=0.5, bins=5, histtype='bar', ec='black', color='red')
ax1.ticklabel_format(useOffset=False)
ax1.xaxis.get_offset_text().set_visible(False)

# Second Plot
ax2.title.set_text('Members of the H. of Rep.')
ax2.title.set_color('blue')
ax2.set_xlabel('Income range - $10^6', color='purple')
ax2.set_ylabel('Frequency', color='purple')
countsR, binsR, barsR = ax2.hist(listNetWorthRepr, alpha=0.5, bins=5, histtype='bar', ec='black', color='blue')
ax2.xaxis.get_offset_text().set_visible(False)

plt.ticklabel_format(useOffset=False)
plt.show()

binsS = np.round(binsS, 0).astype(int)
binsR = np.round(binsR, 0).astype(int)

listSBins = []
for i in range(len(binsS) - 1):
    concatenatedE = binsS.astype('str')[i:i + 2]
    listSBins.append('{1}{0}'.format('-$'.join(concatenatedE), '$'))

figS = go.Figure(data=[go.Table(
    header=dict(values=listSBins,
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=countsS,
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

figS.show()

listRBins = []
for i in range(len(binsR) - 1):
    concatenatedH = binsR.astype('str')[i:i + 2]
    listRBins.append('{1}{0}'.format('-$'.join(concatenatedH), '$'))

figR = go.Figure(data=[go.Table(
    header=dict(values=listRBins,
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=countsR,
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])

figR.show()

print(listNetWorthSenators.mean())
print(listNetWorthSenators.std())
print(np.ptp(listNetWorthSenators, axis=0))

print(listNetWorthRepr.mean())
print(listNetWorthRepr.std())
print(np.ptp(listNetWorthRepr, axis=0))

print()
