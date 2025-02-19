import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.test1 import AM_PromptIR
from My_loss import CharbonnierLoss,Fu_loss 
import lightning.pytorch as pl
import torch.nn.functional as F

class AM_PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        
    
    def forward(self,x,class_name):
        return self.net(x,class_name)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]



def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch,class_name) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch,class_name)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))



def test_Derain(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch,class_name) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch,class_name)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

    return psnr.avg, ssim.avg

def test_Dehaze(net, dataset, task="dehaze"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch,class_name) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch,class_name)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    return psnr.avg, ssim.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="/project/train/src_repo/PromptIR-main/data/data/test/denoise1", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="/project/train/src_repo/PromptIR-main/data/data/test/derain", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="/project/train/src_repo/PromptIR-main/data/data/test/dehaze", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="/project/train/src_repo/PromptIR-main/output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="/project/train/src_repo/PromptIR-main/best_result/test_GSFN_ciajiann/epoch=116-step=529893-v1.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = testopt.ckpt_name

    # files = []
    # for filename in os.listdir(ckpt_path):
    #     file_path = os.path.join(ckpt_path, filename)
    #     if os.path.isfile(file_path):
    #         files.append(file_path)
    # files_out = files[-40:]
    
    denoise_splits = ["low/"]
    derain_splits = ["Rain100L/"]
    dehaze_splits = ["SOTS/"]

    denoise_tests = []
    derain_tests = []
    dehaze_tests = []
    best_rain_ssim=0
    best_rain_psnr=0
    best_haze_ssim=0
    best_haze_psnr=0
    best_rain_ssim_1=''
    best_rain_psnr_1=''
    best_haze_ssim_1=''
    best_haze_psnr_1=''

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)


    print("CKPT name : {}".format(ckpt_path))

    net  = AM_PromptIRModel().load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    
    if testopt.mode == 0:
        for testset,name in zip(denoise_tests,denoise_splits) :
            
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(opt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    elif testopt.mode == 3:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)
#             print('Start {} testing Sigma=15...'.format(name))
#             test_Denoise(net, testset, sigma=15)

#             print('Start {} testing Sigma=25...'.format(name))
#             test_Denoise(net, testset, sigma=25)

#             print('Start {} testing Sigma=50...'.format(name))
#             test_Denoise(net, testset, sigma=50)



#         derain_base_path = testopt.derain_path
#         print(derain_splits)
#         for name in derain_splits:

#             print('Start testing {} rain streak removal...'.format(name))
#             testopt.derain_path = os.path.join(derain_base_path,name)
#             derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
#             # psnr,ssim = test_Derain(net, derain_set, task="derain")
#             for ckpt_path in files_out:
#                 net  = AM_PromptIRModel().load_from_checkpoint(ckpt_path).cuda()
#                 net.eval()
#                 psnr,ssim = test_Derain(net, derain_set, task="derain")
#                 if best_rain_psnr<psnr:
#                     best_rain_psnr = psnr
#                     best_rain_psnr_1 = ckpt_path
#                 if best_rain_ssim<ssim:
#                     best_rain_ssim = ssim
#                     best_rain_ssim_1=ckpt_path


#         print('Start testing SOTS...')
#         # test_Derain_Dehaze(net, derain_set, task="dehaze")
#         dehaze_base_path = testopt.dehaze_path
#         print(dehaze_splits)
#         for name in dehaze_splits:

#             print('Start testing {} rain streak removal...'.format(name))
#             testopt.dehaze_path = os.path.join(dehaze_base_path,name)
#             dehaze_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
#             # psnr,ssim = test_Dehaze(net, dehaze_set, task="dehaze")
#             for ckpt_path in files_out:
#                 net  = AM_PromptIRModel().load_from_checkpoint(ckpt_path).cuda()
#                 net.eval()
#                 psnr ,ssim = test_Dehaze(net, dehaze_set, task="dehaze")
#                 if best_haze_psnr < psnr:
#                     best_haze_psnr = psnr
#                     best_haze_psnr_1 = ckpt_path
#                 if best_haze_ssim< ssim:
#                     best_haze_ssim = ssim
#                     best_haze_ssim_1 = ckpt_path
#         print("heze_PSNR: %.2f, haze_SSIM: %.4f" % (best_haze_psnr, best_haze_ssim))
#         print('best_haze_psnr_net {}'.format(best_haze_psnr_1))
#         print('best_haze_ssim_net{} '.format(best_haze_ssim_1))
#         print("rain_PSNR: %.2f, rain_SSIM: %.4f" % (best_rain_psnr, best_rain_ssim))
#         print('best_rain_psnr_net{} '.format(best_rain_psnr_1))
#         print('best_rain_ssim_net {}'.format(best_rain_ssim_1))
