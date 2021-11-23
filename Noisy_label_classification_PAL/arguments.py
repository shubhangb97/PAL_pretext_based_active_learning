import argparse
import os

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset used - cifar10 | cifar100 | svhn ')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size used for training and testing')
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--data_path', type=str, default='../data', help='Path to where the data is')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')

    parser.add_argument('--optim_task', type=str, default='sgd', help='Optimizer (adam|sgd)')
    parser.add_argument('--scheduler_task', type=str, default='decay_step', help='Scheduler (cosine|decay_step)')
    parser.add_argument('--lr_task', type=float, default=0.01, help='Task LR')

    parser.add_argument('--optim_Rot', type=str, default='sgd', help='Optimizer (adam|sgd)')
    parser.add_argument('--scheduler_Rot', type=str, default='decay_step', help='Scheduler (cosine|decay_step|none)')
    parser.add_argument('--lr_rot', type=float, default=0.01, help=' Scoring network LR')
    parser.add_argument('--rot_train_epochs', type=int, default=100, help='training epochs for Scoring network')

    parser.add_argument('--lambda_kl', type=float, default=1, help='Weight for KL based score')
    parser.add_argument('--train_loss_weightClassif', type=float, default=1, help='Scoring network train loss weight for classif_loss')
    parser.add_argument('--val_loss_weightClassif', type=float, default=1, help='Scoring network validation loss weight for classif_loss')


    parser.add_argument('--lambda_rot', type=float, default=1, help='weight for self-supervised Rot score')
    parser.add_argument('--val_loss_weightRotation', type=float, default=1, help='Scoring net validation loss weight for rot_loss')
    parser.add_argument('--train_loss_weightRotation', type=float, default=1, help='Scoring net train loss weight for rot_loss')

    parser.add_argument('--valtype',type=str,default='loss',help='validation type (loss | accuracy)')

    parser.add_argument('--noise_level',type=float,default=0.2,help='Fraction / chance that label is noisy')
    parser.add_argument('--samplebatch_size',type=int,default=512,help='Sub query batch size')
    parser.add_argument('--lambda_div', type=float, default=0, help='Weight for diversity')

    args = parser.parse_args()



    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    args.log_name = "accexp"+str(args.lr_rot) + '_' + str(args.lr_task) + '_' + str(args.lambda_kl) + '_'+str(args.lambda_rot) + '_' +  args.optim_task + args.log_name
    return args
