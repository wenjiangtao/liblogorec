# -*- coding: utf-8 -*-
import sys,os
import glob
dl = glob.glob(sys.argv[1]+'/*')
mlist = {}
thrs = {}
desd = {}
toti = 0
for d in dl:
    fl = glob.glob(d+'/*.jpg') + glob.glob(d+'/*.JPG')
    tpls = glob.glob(d+'/template/*.jpg') + glob.glob(d+'/template/*.png') + glob.glob(d+'/template/*.jpeg')
    tpl = tpls[0]
    print 'matching',tpl,'with',fl
    mlist[tpl] = fl
    toti += len(fl)
OF = open(sys.argv[3],'w')
print >>OF, len(mlist),toti
for k,tpl in enumerate(mlist):
    print >>OF, sys.argv[2] + '/'+ str(k) +'.txt'
for k,tpl in enumerate(mlist):
    fl = mlist[tpl]
    for im in fl:
        print >>OF, im, k
OF.close()
