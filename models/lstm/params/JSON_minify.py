
import ast, sys
def main(filename): # assumes it's in this directory
    f = open(filename,'r')
    print f

    json = ''
    eof = False
    while eof==False:
        line = f.readline()
        print line
        if line==[]: eof==True; break
        ex = line.find('\n')
        print ex
        if ex==-1: print "invalid parameter file..."
        cx = line.find('#')
        print cx
        if cx==-1:
            json += line[0:(ex-1)]
        else:
            json += line[0:(cx-1)]

    dict = ast.literal_eval(json)
    print dict
    return dict

if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)