import sys

if __name__ == '__main__':
    with open(sys.argv[1]) as f:
	header = f.readline()
	print header
	c = -1
	d = 1
	for line in f.readlines():
	    j = int(line.split(',')[0])
	    if j - c > d:
		print j
	    c = j

