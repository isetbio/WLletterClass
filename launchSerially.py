# -*- coding: utf-8 -*-
# MBP:
# sceneLights = ['5', '15', '45'];
# dists = ['140']
# loop = '10'


# # sienna:
sceneLights = ['1'];
dists = ['25', '50', '70','100', '140']
loop = '5000'


import subprocess as sp


for s in sceneLights:
    for d in dists:
        cmd = str('python scienStanfordPoster.py ' +
                  s + ' ' + d + ' ' + loop +
                  ' >> ' + s + '_' + d + '_' + loop + '.txt'
                  )
        # print cmd
        spcmd = sp.call(cmd, shell=True)
