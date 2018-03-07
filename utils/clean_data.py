import sys
import os


def main(argv):

    try:
        filename = argv[1]
    except:
        print('you need to supply a filename as an argument.')
        exit()

    buff = str()

    with open(filename) as data_in:
        line = data_in.readline()
        while len(line):
            # clear lines that are just '\n'
            while line == '\n':
                line = data_in.readline()

            # Remove lines that start with ' - '
            if line[:3].find(' - ') != -1:
                line = line[3:]

            # Remove spaces at start of line
            while line[0] == ' ':
                line = line[1:]

            # Create new lines that have ' - ' in them
            while line.find(' - ') != -1:
                line = line.replace(' - ', '\n')

            # Remove any '- ' in a line.
            while line.find('- ') != -1:
                line = line.replace('- ', '')

            # if there are multiple sentences on a line, split them
            while line.find('?! ') != -1:
                line = line.replace('?! ', '?!\n')

            while line.find('!? ') != -1:
                line = line.replace('!? ', '!?\n')

            while line.find('? ') != -1:
                line = line.replace('? ', '?\n')

            while line.find('! ') != -1:
                line = line.replace('! ', '!\n')

            while line.find('. ') != -1:
                line = line.replace('. ', '.\n')

            line = line.lower()

            # Correction to lines ending in Mr. Mrs. Ms. Dr.
            if line[-4:] == 'mr.\n':
                append = data_in.readline()
                line = line.rstrip()
                line += ' ' + append
                line = line.replace('mr. ', 'Mr. ')
            elif line[-5:] == 'mrs.\n':
                append = data_in.readline()
                line = line.rstrip()
                line += ' ' + append
                line = line.replace('mrs. ', 'Mrs. ')
            elif line[-4:] == 'ms.\n':
                append = data_in.readline()
                line = line.rstrip()
                line += ' ' + append
                line = line.replace('ms. ', 'Ms. ')
            elif line[-4:] == 'dr.\n':
                append = data_in.readline()
                line = line.rstrip()
                line += ' ' + append
                line = line.replace('dr. ', 'Dr. ')
            else:
                pass

            buff += line

            line = data_in.readline()
            while line == '\n':
                line = data_in.readline()




    with open('data_out.txt', 'w') as data_out:
        buff = "<start> " + buff
        buff = buff.replace('\n', ' <end>\n<start> ') # No punctuation
        buff = buff.replace('.\n', '. <end>\n<start> ')
        buff = buff.replace('?\n', '? <end>\n<start> ')
        buff = buff.replace('!\n', '! <end>\n<start> ')
        buff = buff.replace('?!\n', '?! <end>\n<start> ')
        buff = buff.replace('!?\n', '!? <end>\n<start> ')

        buff = buff[:-8]

        data_out.write(buff)

    with open('data_out.txt', 'r+') as verify:

        line = verify.readline()
        ln = 1

        while len(line) != 0:
            if line[-8:] != '. <end>\n' and\
                line[-8:] != '! <end>\n' and\
                line[-8:] != '? <end>\n' and\
                line[-9:] != '?! <end>\n' and\
                line[-9:] != '!? <end>\n':

                    print("Warn: line", ln, "doesn't end with punctuation:", line)

            line = verify.readline()



            ln += 1











main(sys.argv)
