import sys
import os
import re



def extract_quotes(para):

    to_return = []
    i = 1
    for line in para:
        if line.find('"') != -1:
            extended = line.split('"')
            for item in extended:
                to_return.insert(i, item)
                i += 1
        else:
            to_return.insert(i, line)
            i += 1

    return to_return


def clear_whitespace(line):
    # clear lines that are just '\n'
    if line == '\n' or line == ' ':
        return ''

    line = line.replace('\n', '')
    line = line.replace('\t', '')

    while line.find('  ') != -1:
        line = line.replace('  ', ' ')

    return line


def join_prefixes(lst):

    i = 0
    while i < len(lst) and len(lst) > 1:
        if len(lst[i]) > 6 and len(lst[i+1]) > 1:
            if len(lst) > 1 and lst[i][-5:] == ' Mr.\n':
                lst[i:i+2] = [' '.join(lst[i:i+2])]
                i+= 1

            if len(lst) > 1 and lst[i][-5:] == ' mr.\n':
                lst[i:i+2] = [' '.join(lst[i:i+2])]
                i+= 1

            if len(lst) > 1 and lst[i][-6:] == ' Mrs.\n':
                lst[i:i+2] = [' '.join(lst[i:i+2])]
                i+= 1

            if len(lst) > 1 and lst[i][-6:] == ' mrs.\n':
                lst[i:i+2] = [' '.join(lst[i:i+2])]
                i+= 1

            if len(lst) > 1 and lst[i][-5:] == ' Ms.\n' and len(lst) > 1:
                lst[i:i+2] = [' '.join(lst[i:i+2])]
                i+= 1

            if len(lst) > 1 and lst[i][-5:] == ' ms.\n' and len(lst) > 1:
                lst[i:i+2] = [' '.join(lst[i:i+2])]
                i+= 1

            if len(lst) > 1 and lst[i][-5:] == ' Dr.\n' and len(lst) > 1:
                lst[i:i+2] = [' '.join(lst[i:i+2])]
                i+= 1

            if len(lst) > 1 and lst[i][-5:] == ' dr.\n' and len(lst) > 1:
                lst[i:i+2] = [' '.join(lst[i:i+2])]
                i+= 1
        i+=1

    return lst


def format_front(line):
    while len(line) > 0:
        if line[0] == ' ' or line[0] == '-':
            line = line[1:]
        else:
            break

    return line


def main(argv):

    try:
        filename = argv[1]
        fileout = argv[2]
    except:
        print('you need to supply a filename as an argument.')
        exit()

    buff = str()
    with open(filename, 'r') as data_in:
        line = data_in.readline()
        while len(line):
            para = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', line)
            para = extract_quotes(para)
            para = list(filter(lambda a: a != '', para))

            while len(para) == 1:
                another = data_in.readline()
                another = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', another)
                para = [line] + another
                para = list(filter(lambda a: a != '', para))

            para = join_prefixes(para)
            for l in range(len(para)):
                line = para[l]

                line = line.lower()
                line = clear_whitespace(line)
                line = format_front(line)
                line = '<start> ' + line + ' <end>\n'

                if len(line) > 15:
                    buff += line
            line = data_in.readline()

    with open(fileout, 'w') as data_out:
        data_out.write(buff)

main(sys.argv)
