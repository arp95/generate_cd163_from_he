"""
Author: Arpit Aggarwal
Summary: Model class
"""


# header files
import torch
from .base_model import BaseModel
from . import networks

class Pix2PixModel(BaseModel):
    """ This class implements the DeepLIIF model, for learning a mapping from input images to modalities given paired data."""

    def __init__(self, opt):
        """Initialize the DeepLIIF class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # loss weights in calculating the final loss
        self.loss_G_weights = [0.8, 0.1, 0.1]
        self.loss_D_weights = [0.8, 0.1, 0.1]
        self.loss_names = []
        self.visual_names = ['real_A', 'fake_B_1', 'real_B', 'fake_B_2', 'real_C', 'fake_B_3', 'real_D']
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        for i in range(1, 4):
            self.loss_names.extend(['G_GAN_' + str(i), 'G_L1_' + str(i), 'D_real_' + str(i), 'D_fake_' + str(i)])

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.opt.phase == "train":
            self.model_names = []
            for i in range(1, 4):
                self.model_names.extend(['G' + str(i), 'D' + str(i)])
        else:  # during test time, only load G
            self.model_names = []
            for i in range(1, 4):
                self.model_names.extend(['G' + str(i)])

        # define networks (both generator and discriminator)
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet_256', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.opt.phase == 'train':  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD1 = networks.define_D(opt.input_nc+opt.output_nc , opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD3 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.opt.phase == 'train':
            # define loss functions
            self.criterionGAN_BCE = networks.GANLoss('vanilla').to(self.device)
            self.criterionGAN_lsgan = networks.GANLoss('lsgan').to(self.device)
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            params = list(self.netG1.parameters()) + list(self.netG2.parameters()) + list(self.netG3.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.netD1.parameters()) + list(self.netD2.parameters()) + list(self.netD3.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param input (dict): include the input image and the output modalities
        """

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_D = input['D'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B_1 = self.netG1(self.real_A)   # cd163 ihc image generator
        self.fake_B_2 = self.netG2(self.real_A)   # nuclei mask image generator
        self.fake_B_3 = self.netG3(self.real_A)   # brown mask image generator

    def backward_D(self):
        """Calculate GAN loss for the discriminators"""
        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)  # Conditional GANs; feed h&e input and cd163 ihc output to the discriminator
        fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)  # Conditional GANs; feed h&e input and nuclei mask output to the discriminator
        fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)  # Conditional GANs; feed h&e input and cd163 brown mask output to the discriminator

        pred_fake_1 = self.netD1(fake_AB_1.detach())
        pred_fake_2 = self.netD2(fake_AB_2.detach())
        pred_fake_3 = self.netD2(fake_AB_3.detach())

        self.loss_D_fake_1 = self.criterionGAN_lsgan(pred_fake_1, False)
        self.loss_D_fake_2 = self.criterionGAN_lsgan(pred_fake_2, False)
        self.loss_D_fake_3 = self.criterionGAN_lsgan(pred_fake_3, False)

        real_AB_1 = torch.cat((self.real_A, self.real_B), 1)
        real_AB_2 = torch.cat((self.real_A, self.real_C), 1)
        real_AB_3 = torch.cat((self.real_A, self.real_D), 1)

        pred_real_1 = self.netD1(real_AB_1)
        pred_real_2 = self.netD2(real_AB_2)
        pred_real_3 = self.netD2(real_AB_3)

        self.loss_D_real_1 = self.criterionGAN_lsgan(pred_real_1, True)
        self.loss_D_real_2 = self.criterionGAN_lsgan(pred_real_2, True)
        self.loss_D_real_3 = self.criterionGAN_lsgan(pred_real_3, True)

        # combine losses and calculate gradients
        self.loss_D = (self.loss_D_fake_1 + self.loss_D_real_1) * 0.8 * self.loss_D_weights[0] + \
                      (self.loss_D_fake_2 + self.loss_D_real_2) * 0.1 * self.loss_D_weights[1] + \
                      (self.loss_D_fake_3 + self.loss_D_real_3) * 0.1 * self.loss_D_weights[2]

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)
        fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)
        fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)

        pred_fake_1 = self.netD1(fake_AB_1)
        pred_fake_2 = self.netD2(fake_AB_2)
        pred_fake_3 = self.netD2(fake_AB_3)

        self.loss_G_GAN_1 = self.criterionGAN_lsgan(pred_fake_1, True)
        self.loss_G_GAN_2 = self.criterionGAN_lsgan(pred_fake_2, True)
        self.loss_G_GAN_3 = self.criterionGAN_lsgan(pred_fake_3, True)

        # Second, G(A) = B
        self.loss_G_L1_1 = self.criterionSmoothL1(self.fake_B_1, self.real_B) * self.opt.lambda_L1
        self.loss_G_L1_2 = self.criterionSmoothL1(self.fake_B_2, self.real_C) * self.opt.lambda_L1
        self.loss_G_L1_3 = self.criterionSmoothL1(self.fake_B_3, self.real_D) * self.opt.lambda_L1

        self.loss_G = (self.loss_G_GAN_1 + self.loss_G_L1_1) * self.loss_G_weights[0] + \
                      (self.loss_G_GAN_2 + self.loss_G_L1_2) * self.loss_G_weights[1] + \
                      (self.loss_G_GAN_3 + self.loss_G_L1_3) * self.loss_G_weights[2]
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD1, True)  # enable backprop for D1
        self.set_requires_grad(self.netD2, True)  # enable backprop for D2
        self.set_requires_grad(self.netD3, True)  # enable backprop for D3

        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD1, False)  # D1 requires no gradients when optimizing G1
        self.set_requires_grad(self.netD2, False)  # D2 requires no gradients when optimizing G2
        self.set_requires_grad(self.netD3, False)  # D2 requires no gradients when optimizing G3

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
