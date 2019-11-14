import pandas as pd
import numpy as np

df = pd.read_csv('/Users/andyliu/Downloads/NSDUH_2016_Tab.csv', usecols=['CIGEVER',
'SMKLSSEVR',
'CIGAREVR',
'PIPEVER',
'ALCEVER',
'MJEVER',
'HEREVER',
'PNRNMLIF',
'HEALTH',
'DIFFHEAR',
'DIFFSEE',
'DIFFTHINK',
'DIFFWALK',
'DIFFDRESS',
'DIFFERAND',
'GOVTPROG',
'INCOME',
'IRINSUR4',
'IRWRKSTAT18',
'SEXIDENT',
'IRSEX',
'NEWRACE2',
'MILTFAMLY',
'IRHHSIZ2',
'NRCH17_2',
'IMOTHER',
'IFATHER',
'IRHH65_2',
'YEPCHKHW',
'YEPHLPHW',
'YEPCHORE',
'YEPLMTTV',
'YEPLMTSN',
'YEYARGUP',
'YEYFGTSW',
'YEYFGTGP',
'YEYHGUN',
'SNYATTAK',
'SNYSTOLE',
'YESTSCIG',
'YESTSMJ',
'YESTSALC',
'YESTSDNK',
'YESCHFLT',
'YESCHWRK',
'YELSTGRD',
'IRWRKSTAT',
'K6SCYR',
'SPDYR',
'YMDELT',
'YSDSOVRL',
'AGE2',])

df['NICEVR'] = ((df['SMKLSSEVR'] == 1) | (df['CIGAREVR'] == 1) | (df['PIPEVER'] == 1) | (df['CIGEVER'] == 1)).astype(int)
df['OPEVER'] = ((df['HEREVER'] == 1) | (df['PNRNMLIF'] == 1)).astype(int)
df = df.query('AGE2<7')
df = df.replace(to_replace=['85','94','97','98','99'], value=np.NaN)

df.to_csv('drugdata_students_final.csv')


