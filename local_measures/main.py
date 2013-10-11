import nibabel as nib
import numpy as np
from numpy.linalg import norm
from scipy import stats
from math import *
import sys, glob, xlwt, os.path, re


### load ground-truth directions
################################
print "-> opening ground-truth...",

niiGT = nib.load('ground-truth/peaks.nii.gz')
niiGT_hdr = niiGT.get_header()
niiGT_img = niiGT.get_data()

niiGT_dim = niiGT_hdr.get_data_shape()

nx = niiGT_dim[0]
ny = niiGT_dim[1]
nz = niiGT_dim[2]

print "[OK]\n"


### load white-matter mask
##########################
print "-> opening white-matter mask...",

niiWM = nib.load('ground-truth/fibers.nii.gz')
niiWM_hdr = niiWM.get_header()
niiWM_img = niiWM.get_data()

niiWM_dim = niiWM_hdr.get_data_shape()

niiWM_idx = niiWM_img==1

print "[OK]\n"


### load reconstructions
########################
SUBMISSIONs = glob.glob( "submissions/*.nii.gz" )

# prepare EXCEL output
XLS = xlwt.Workbook()
XLS_sheet = XLS.add_sheet("Local measures")

XLS_sheet.write( 0,  0, "appModel" )
XLS_sheet.write( 0,  1, "acq" )
XLS_sheet.write( 0,  2, "model" )
XLS_sheet.write( 0,  3, "snr" )
XLS_sheet.write( 0,  4, "Pd, mean" )
XLS_sheet.write( 0,  5, "Pd, std" )
XLS_sheet.write( 0,  6, "Pd, min" )
XLS_sheet.write( 0,  7, "Pd, 25 perc" )
XLS_sheet.write( 0,  8, "Pd, 50 perc" )
XLS_sheet.write( 0,  9, "Pd, 75 perc" )
XLS_sheet.write( 0, 10, "Pd, max" )
XLS_sheet.write( 0, 11, "n-, mean" )
XLS_sheet.write( 0, 12, "n-, std" )
XLS_sheet.write( 0, 13, "n-, min" )
XLS_sheet.write( 0, 14, "n-, 25 perc" )
XLS_sheet.write( 0, 15, "n-, 50 perc" )
XLS_sheet.write( 0, 16, "n-, 75 perc" )
XLS_sheet.write( 0, 17, "n-, max" )
XLS_sheet.write( 0, 18, "n+, mean" )
XLS_sheet.write( 0, 19, "n+, std" )
XLS_sheet.write( 0, 20, "n+, min" )
XLS_sheet.write( 0, 21, "n+, 25 perc" )
XLS_sheet.write( 0, 22, "n+, 50 perc" )
XLS_sheet.write( 0, 23, "n+, 75 perc" )
XLS_sheet.write( 0, 24, "n+, max" )
XLS_sheet.write( 0, 25, "AE, mean" )
XLS_sheet.write( 0, 26, "AE, std" )
XLS_sheet.write( 0, 27, "AE, min" )
XLS_sheet.write( 0, 28, "AE, 25 perc" )
XLS_sheet.write( 0, 29, "AE, 50 perc" )
XLS_sheet.write( 0, 30, "AE, 75 perc" )
XLS_sheet.write( 0, 31, "AE, max" )


XLS_row = 1
for filename in SUBMISSIONs:
    print "-> %s" % os.path.basename( filename )

    print "\t* opening data...",

    niiRECON = nib.load( filename )

    niiRECON_hdr = niiRECON.get_header()
    niiRECON_img = niiRECON.get_data()

    niiRECON_dim = niiRECON_hdr.get_data_shape()

    print "[OK]"


    ### check consistency
    print "\t* checking consistency...",

    if len(niiRECON_dim) != len(niiGT_dim) :
        raise Exception("The shape of GROUND-TRUTH and TEAM's RECONSTRUCTION do not match")
    if niiRECON_dim != niiGT_dim :
        raise Exception("'dim' of GROUND-TRUTH and TEAM's RECONSTRUCTION do not match")

    print "[OK]"


    ### compute local metrics
    print "\t* computing local metrics...",
    sys.stdout.flush()

    Pd = np.zeros( niiGT_dim[0:3] )
    nP = np.zeros( niiGT_dim[0:3] )
    nM = np.zeros( niiGT_dim[0:3] )
    AE = np.zeros( niiGT_dim[0:3] )

    for z in range(0,nz):
        for y in range(0,ny):
            for x in range(0,nx):
                if niiWM_img[x,y,z] == 0 :
                    continue

                # NUMBER OF FIBER POPULATIONS
                #############################

                DIR_true = np.zeros( (3,5) )
                DIR_est  = np.zeros( (3,5) )

                # compute M_true, DIR_true, M_est, DIR_est
                M_true = 0
                for d in range(5) :
                    dir = niiGT_img[x,y,z,range(d*3, d*3+3)]
                    f = norm( dir )
                    if f > 0 :
                        DIR_true[:,M_true] = dir / f
                        M_true += 1
                if M_true == 0 :
                    niiWM_img[x,y,z] = 0 # do not consider this voxel in the final score
                    continue    # no fiber compartments found in the voxel

                M_est = 0
                for d in range(5) :
                    dir = niiRECON_img[x,y,z,range(d*3, d*3+3)]
                    f = norm( dir )
                    if f > 0 :
                        DIR_est[:,M_est] = dir / f
                        M_est += 1

                # compute Pd, nM and nP
                M_diff = M_true - M_est
                Pd[x,y,z] = 100 * abs(M_diff) / M_true
                if  M_diff > 0 :
                    nM[x,y,z] = M_diff;
                else :
                    nP[x,y,z] = -M_diff

                # ANGULAR ACCURACY
                ##################

                # precompute matrix with angular errors among all estimated and true fibers
                A = np.zeros( (M_true, M_est) )
                for i in range(0,M_true) :
                    for j in range(0,M_est) :
                        err = acos( min( 1.0, abs(np.dot( DIR_true[:,i], DIR_est[:,j] )) ) ) # crop to 1 for internal precision
                        A[i,j] = min( err, pi-err) / pi * 180;

                # compute the "base" error
                M = min(M_true,M_est)
                err = np.zeros( M )
                notUsed_true = np.array( range(0,M_true) )
                notUsed_est  = np.array(range(0,M_est) )
                AA = np.copy( A )
                for i in range(0,M) :
                    err[i] = np.min( AA )
                    r, c = np.nonzero( AA==err[i] )
                    AA[r[0],:] = float('Inf')
                    AA[:,c[0]] = float('Inf')
                    notUsed_true = notUsed_true[ notUsed_true != r[0] ]
                    notUsed_est  = notUsed_est[  notUsed_est  != c[0] ]

                # account for OVER-ESTIMATES
                if M_true < M_est :
                    if M_true > 0:
                        for i in notUsed_est :
                            err = np.append( err, min( A[:,i] ) )
                    else :
                        err = np.append( err, 45 )
                # account for UNDER-ESTIMATES
                elif M_true > M_est :
                    if M_est > 0:
                        for i in notUsed_true :
                            err = np.append( err, min( A[i,:] ) )
                    else :
                        err = np.append( err, 45 )

                AE[x,y,z] = np.mean( err )

    print "[OK]"


    # output to screen
    print "\t\tPd = %.2f%%" % np.mean( Pd[niiWM_idx] )
    print "\t\tn- = %.3f" % np.mean( nM[niiWM_idx] )
    print "\t\tn+ = %.3f" % np.mean( nP[niiWM_idx] )
    print "\t\tAE = %.2f degree" % np.mean( AE[niiWM_idx] )
    print " "

    # write to EXCEL file
#     m = re.search( "ISBI2013_acq-(\w+)_appModel-(\d+)_model-(\w+)_format-peaks_snr-(\d+).*", 
#                    os.path.basename(filename) )

#     m = re.search( "local_masures.xls", os.path.basename(filename))

#     XLS_sheet.write( XLS_row,  0, int(m.group(2)) )
#     XLS_sheet.write( XLS_row,  1, m.group(1) )
#     XLS_sheet.write( XLS_row,  2, m.group(3) )
#     XLS_sheet.write( XLS_row,  3, int(m.group(4)) )

#     values = Pd[niiWM_idx]
#     XLS_sheet.write( XLS_row,  4, np.mean(values) )
#     XLS_sheet.write( XLS_row,  5, np.std(values) )
#     XLS_sheet.write( XLS_row,  6, np.min(values) )
#     XLS_sheet.write( XLS_row,  7, stats.scoreatpercentile(values,25) )
#     XLS_sheet.write( XLS_row,  8, np.median(values) )
#     XLS_sheet.write( XLS_row,  9, stats.scoreatpercentile(values,75) )
#     XLS_sheet.write( XLS_row, 10, np.max(values) )

#     values = nM[niiWM_idx]
#     XLS_sheet.write( XLS_row, 11, np.mean(values) )
#     XLS_sheet.write( XLS_row, 12, np.std(values) )
#     XLS_sheet.write( XLS_row, 13, np.min(values) )
#     XLS_sheet.write( XLS_row, 14, stats.scoreatpercentile(values,25) )
#     XLS_sheet.write( XLS_row, 15, np.median(values) )
#     XLS_sheet.write( XLS_row, 16, stats.scoreatpercentile(values,75) )
#     XLS_sheet.write( XLS_row, 17, np.max(values) )

#     values = nP[niiWM_idx]
#     XLS_sheet.write( XLS_row, 18, np.mean(values) )
#     XLS_sheet.write( XLS_row, 19, np.std(values) )
#     XLS_sheet.write( XLS_row, 20, np.min(values) )
#     XLS_sheet.write( XLS_row, 21, stats.scoreatpercentile(values,25) )
#     XLS_sheet.write( XLS_row, 22, np.median(values) )
#     XLS_sheet.write( XLS_row, 23, stats.scoreatpercentile(values,75) )
#     XLS_sheet.write( XLS_row, 24, np.max(values) )

#     values = AE[niiWM_idx]
#     XLS_sheet.write( XLS_row, 25, np.mean(values) )
#     XLS_sheet.write( XLS_row, 26, np.std(values) )
#     XLS_sheet.write( XLS_row, 27, np.min(values) )
#     XLS_sheet.write( XLS_row, 28, stats.scoreatpercentile(values,25) )
#     XLS_sheet.write( XLS_row, 29, np.median(values) )
#     XLS_sheet.write( XLS_row, 30, stats.scoreatpercentile(values,75) )
#     XLS_sheet.write( XLS_row, 31, np.max(values) )

#     XLS_row += 1


#XLS.save("local_masures.xls")
