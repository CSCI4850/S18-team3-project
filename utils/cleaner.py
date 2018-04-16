import sys
import re


def has_prefix(line):
    if len(line) >= 5:
        return (line[-5:] == ' Mr.\n' or\
                line[-5:] == ' mr.\n' or\
                line[-5:] == ' Ms.\n' or\
                line[-5:] == ' ms.\n' or\
                line[-5:] == ' Dr.\n')

    if len(line) >= 6:
        return (line[-6:] == ' Mrs.\n' or\
                line[-6:] == ' mrs.\n')


def join(seq, line):
    # split the line to append into sentences
    line = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', line)

    # connect the last in seq to the first in line
    seq[-1] = seq[-1].rstrip('\n') + ' ' + line[0]

    # remove the one connected
    line.remove(line[0])

    # append the line for further proccesing
    seq += line

    # sometimes an empty string appears.
    if seq[-1] == '':
        seq = seq[:-1]

    return seq


def join_prefixes(seq):
    i = 1

    while i < len(seq) + 1:
        if has_prefix(seq[i-1]) and i < len(seq):
            seq[i-1:i+1] = [''.join(seq[i-1:i+1])]
        i += 1

    for i in range(len(seq)):
        seq[i] = clear_whitespace(seq[i])

    return seq


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

    if line[0:2] == '  ':
        line = line[2:]

    return line


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
            seq = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', line)
            seq = extract_quotes(seq)
            seq = list(filter(lambda a: a != '', seq))

            while has_prefix(seq[-1]):
                seq = join(seq, data_in.readline())

            for l in range(len(seq)):
                line = seq[l]

                line = line.lower()
                line = clear_whitespace(line)
                line = format_front(line)
                line = 'ST ' + line + ' EN\n'

                if len(line) > 7:
                    buff += line
            line = data_in.readline()

    with open(fileout, 'w') as data_out:
        data_out.write(buff)

if __name__ == '__main__':
    main(sys.argv)
