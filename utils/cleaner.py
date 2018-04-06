import sys
import os
import re


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

            while line.find('--') != -1:
                line = line.replace('--', '')

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

            line = line.replace('  ', ' ')

            # Correction to lines ending in Mr. Mrs. Ms. Dr.
            if line[-5:] == ' mr.\n':
                append = data_in.readline()
                line = line.rstrip()
                line += ' ' + append
            elif line[-6:] == ' mrs.\n':
                append = data_in.readline()
                line = line.rstrip()
                line += ' ' + append
            elif line[-5:] == ' ms.\n':
                append = data_in.readline()
                line = line.rstrip()
                line += ' ' + append
            elif line[-5:] == ' dr.\n':
                append = data_in.readline()
                line = line.rstrip()
                line += ' ' + append
            else:
                pass

            line = line.replace('. " ', '."' )
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

        buff = buff.replace('<start> <end>\n', '')
        buff = buff.replace('<start> - <end>\n', '')


        data_out.write(buff)

    # with open('data_out.txt', 'r+') as data:
    #     buff = data.read()
    #     with open('../data/male.txt', 'r') as names:
    #         name = names.readline()
    #         name = name.rstrip()
    #         while len(name):
    #             # note '?!' and '!?' are caught by '?' and '!'
    #             buff = buff.replace(' ' + name.lower() + ' ', ' ' + name + ' ')
    #             buff = buff.replace(' ' + name.lower() + ',', ' ' + name + ',')
    #             buff = buff.replace(' ' + name.lower() + '.', ' ' + name + '.')
    #             buff = buff.replace(' ' + name.lower() + ':', ' ' + name + ':')
    #             buff = buff.replace(' ' + name.lower() + ';', ' ' + name + ';')
    #             buff = buff.replace(' ' + name.lower() + '!', ' ' + name + '!')
    #             buff = buff.replace(' ' + name.lower() + '?', ' ' + name + '?')
    #             buff = buff.replace(' ' + name.lower() + '"', ' ' + name + '"')
    #             buff = buff.replace('"' + name.lower() + ' ', '"' + name + ' ')
    #             buff = buff.replace(' ' + name.lower() + "'s", ' ' + name + "'s")
    #
    #             name = names.readline()
    #             name = name.rstrip()
    #
    #     with open('../data/female.txt', 'r') as names:
    #         name = names.readline()
    #         name = name.rstrip()
    #         while len(name):
    #             # note '?!' and '!?' are caught by '?' and '!'
    #             buff = buff.replace(' ' + name.lower() + ' ', ' ' + name + ' ')
    #             buff = buff.replace(' ' + name.lower() + ',', ' ' + name + ',')
    #             buff = buff.replace(' ' + name.lower() + '.', ' ' + name + '.')
    #             buff = buff.replace(' ' + name.lower() + ':', ' ' + name + ':')
    #             buff = buff.replace(' ' + name.lower() + ';', ' ' + name + ';')
    #             buff = buff.replace(' ' + name.lower() + '!', ' ' + name + '!')
    #             buff = buff.replace(' ' + name.lower() + '?', ' ' + name + '?')
    #             buff = buff.replace(' ' + name.lower() + '"', ' ' + name + '"')
    #             buff = buff.replace('"' + name.lower() + ' ', '"' + name + ' ')
    #             buff = buff.replace(' ' + name.lower() + "'s", ' ' + name + "'s")
    #
    #             name = names.readline()
    #             name = name.rstrip()
    #
    #     data.write(buff)


main(sys.argv)
