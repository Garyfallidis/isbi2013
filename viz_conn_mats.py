from dipy.io.pickles import load_pickle


def show_conn_mat(filename):

    d = load_pickle(filename)        

    method = filename.split('__')[1].split('_')[0]
    
    subplot(2, 2, 1)

    title(method + ' full')

    imshow(d['mat'])    

    subplot(2, 2, 2)

    title(method + ' 0.5')

    imshow(d['conn_mats'][0])

    subplot(2, 2, 3)

    title(method + ' 1.')

    imshow(d['conn_mats'][1])    

    subplot(2, 2, 4)

    title(method + ' 1.5')

    imshow(d['conn_mats'][2])        

    print 'Diffs: ', d['diffs']

    try:
        print 'Ratio: ', d['ratio']
    except KeyError:
        print 'KeyError: ratio does not exist'    

if __name__ == '__main__':
	
	import sys

	show_conn_mat(sys.argv[1])
