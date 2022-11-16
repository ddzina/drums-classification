import os

PATH = 'Dataset'


def main():
    for dirpath, dirnames, filenames in os.walk(PATH):
        if dirpath is not PATH:
            for i, f in enumerate(filenames):
                os.rename(f'{os.path.join(dirpath, f)}', f'{os.path.join(dirpath, f"{dirpath[8:]} {i+1}.wav")}')


if __name__ == '__main__':
    main()
