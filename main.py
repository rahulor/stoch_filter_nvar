"""
@author: rahulor@live.com
"""
import numpy as np
import os
import shutil
import computation
import view

def clean():
    shutil.rmtree("data", ignore_errors=True)
    os.mkdir("data")
    shutil.rmtree("fig", ignore_errors=True)
    os.mkdir("fig")
    # FileExistsError: [WinError 183] Cannot create a file when that file already exists:
    # < please close data and fig if it is open already>
def document():
    print('Generating pdf ...')
    if os.system("pdflatex doc.tex"):
        print('Error; no pdf output', '\nClose doc.pdf if it is already open')
    else:
        print('Done!\nPlease open doc.pdf')
    os.remove("doc.log")
    os.remove("doc.aux")
    
def write_to_file(time_elapsed, train_error, test_error):
    path_info= 'data/result' + '.txt'
    finfo = open(path_info, 'w')
    print('time_elapsed (train and test)'.ljust(30,' '), ':', np.round(time_elapsed,3), '[s]', file = finfo)
    print('='*20, 'training phase'.center(15,' '), '='*20, file = finfo)
    print(train_error.to_string(), file = finfo)
    print('='*20, 'testing phase'.center(15,' '), '='*20, file=finfo)
    print(test_error.to_string(), file = finfo)
    finfo.close()
def figures():
    view.weights()
    view.dataset()
def main():
    clean()
    time_elapsed, train_error, test_error = computation.run()
    write_to_file(time_elapsed, train_error, test_error)
if __name__ == '__main__':
    main()
    figures()
    document()
    
