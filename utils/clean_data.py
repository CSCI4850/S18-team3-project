import sys


def main(argv):

    try:
        filename = argv[1]
    except:
        print('you need to supply a filename as an argument.')
        exit()

    with open(filename) as data_in:

        line = data_in.readline()
        line = line.rstrip()

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

            # Correction to lines ending in Mr. Mrs. Ms. Dr.
            if line[-3:] == 'Mr.':
                append = data_in.readline()
                line += ' ' + append
                line = line.rstrip()
            elif line[-3:] == 'Ms.':
                append = data_in.readline()
                line += ' ' + append
                line = line.rstrip()
            elif line[-3:] == 'Dr.':
                append = data_in.readline()
                line += ' ' + append
                line = line.rstrip()
            elif line[-4:] == 'Mrs.':
                append = data_in.readline()
                line += ' ' + append
                line = line.rstrip()
            else:
                pass


            print(line)
            line = data_in.readline()
            line = line.rstrip('\n')
    data_in.closed

    #with open(argv[1]) as data_in:
    #    ln = 1
    #
    #    line = data_in.readline()
    #        while len(line):
    #        if
    #data_in.closed




main(sys.argv)
