# -*- coding: utf-8 -*-
import sys,os
import glob
dl = glob.glob(sys.argv[1]+'/*')
mlist = {}
thrs = {}
desd = {}
for d in dl:
    fl = glob.glob(d+'/*.jpg') + glob.glob(d+'/*.JPG')
    tpls = glob.glob(d+'/template/*.jpg') + glob.glob(d+'/template/*.png') + glob.glob(d+'/template/*.jpeg')
    tpl = tpls[0]
    print 'matching',tpl,'with',fl
    mlist[tpl] = fl
OF = open(sys.argv[3],'w')
print >>OF, len(mlist)
for k,tpl in enumerate(mlist):
    print >>OF, sys.argv[2] + '/'+ str(k) +'.txt'
    fl = mlist[tpl]
    print >>OF, len(fl)
    print >>OF, tpl
    for im in fl:
        print >>OF, im
        
OF.close()
