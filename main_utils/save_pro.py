import os
import pickle
import math
import heapq
import csv
import re


def save_progress(args, Accuracy_accountant, grad_norm_accountant ):
    '''
    This function saves our progress either in an existing file structure or writes a new file.
    :param save_dir: STRING: The directory where to save the progress.
    :param model: DICTIONARY: The model that we wish to save.
    :param Delta_accountant: LIST: The list of deltas that we allocared so far.
    :param Accuracy_accountant: LIST: The list of accuracies that we allocated so far.
    :param PrivacyAgent: CLASS INSTANCE: The privacy agent that we used (specifically the m's that we used for Federated training.)
    :param FLAGS: CLASS INSTANCE: The FLAGS passed to the learning procedure.
    :return: nothing
    '''
    save_dir = os.path.join(os.getcwd(), args.save_dir,
                            'res_{}'.format(str(args.dataset)+'_'+str(args.dp_type)
                    +'_'+str(args.clip)+'_'+str(args.gamma)+'_'+str(args.lr)) )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "{}".format( args.time,
                                    ('-sess' if args.sess else '')
                                     )
                                    
    with open(os.path.join(save_dir, filename + '.csv'), "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(Accuracy_accountant)
        writer.writerow(grad_norm_accountant)
        