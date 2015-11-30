sceneLights = ['45'];
dists = ['25']
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
