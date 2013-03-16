from dipy.io.pickles import load_pickle


def show_conn_mat(filename):

    d = load_pickle(filename)        

    title(filename)

    subplot(2, 2, 1)

    imshow(d['mat'])    

    subplot(2, 2, 2)

    imshow(d['conn_mats'][0])

    subplot(2, 2, 3)

    imshow(d['conn_mats'][1])    

    subplot(2, 2, 4)

    imshow(d['conn_mats'][2])        

    print 'Diffs: ', d['diffs']

if __name__ == '__main__':
	
	import sys

	show_conn_mat(sys.argv[1])
