import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=18,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
# parser.add_argument('--de_type', nargs='+', default=['dehaze', 'derain'],
#                     help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='PromptIR-main/data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='PromptIR-main/data/data/Train/denoise',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='PromptIR-main/data/data/Train/derain',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='PromptIR-main/data/data/Train/dehazy',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="results/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="PromptIR-main/ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="PromptIR-main/ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 1,help = "Number of GPUs to use for training")

options = parser.parse_args()

