import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.transforms as transforms
from pytorchyolo import models
import numpy as np
from util.util import Resize, DEFAULT_TRANSFORMS, soft_nms, rescale_boxes


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.yolov3_model = models.load_model(
            "config/yolov3.cfg",
            "yolov3_weights/yolov3_ckpt_150.pth"
        )
        self.yolov3_model.eval() 
        # yolov3_model的模式会影响输出的形状
        for param in self.yolov3_model.parameters():
            param.requires_grad = False
        
        self.anchors = torch.tensor([
            [10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]
        ], dtype=torch.float32)

        self.lamda_D = 1
        self.lamda_C = 10
        self.lamda_T = 5
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.loss_names = ['D_A', 'G_A', 'D_B', 'G_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # visual_names_A = ['real_A', 'fake_B', 'rec_A']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B']
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.A_label = input['A_label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)


    def domain_consistency_loss(self, fake_source_feat, real_source_feat):
        # 如果输入是 [B, C, H, W]，先转成 [B, C]
        def get_mean(feat):
            if feat.dim() > 2:
                feat = feat.mean(dim=[2, 3])  # spatial mean
            return feat.mean(dim=0)  # batch mean

        # 计算中心向量
        mean_fake_S = get_mean(fake_source_feat)
        mean_real_S = get_mean(real_source_feat)

        # L2 范数平方
        loss_S = torch.norm(mean_fake_S + mean_real_S, p=2).pow(2)

        # 近似 sigmoid 形式（数值稳定）
        loss_S = 1 / (1 + torch.exp(-loss_S))

        return loss_S  # scalar tensor


    def content_consistency_loss(self, x_s, G_ST, G_TS):
        # source cycle: x_s → G_ST(x_s) → G_TS(G_ST(x_s))
        s_to_t = G_ST(x_s)
        s_recon = G_TS(s_to_t)
        loss_s = torch.norm(x_s - s_recon, p=2).pow(2)

        # Apply the sigmoid-like transformation
        loss_s = 1 / (1 + torch.exp(-loss_s))
        # 不加这个log反而好点

        return loss_s
    
    def calculate_iou(self, box1, box2):
        """
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
        Returns:
            IoU (float)
        """
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou
    

    def tfl_loss(self, rec_S): # batchsize = 1
        img = rec_S[0]
        img_size = 416
        input_img = transforms.Compose([
            Resize(img_size)])((img, np.zeros((1, 5))))[0].unsqueeze(0)
        detections = self.yolov3_model(input_img) 
        detections = soft_nms(detections, 0.5, 0.5)
        detections = rescale_boxes(detections[0], 416, original_shape=img.shape[1:])  # shape: [N, 6] (x1, y1, x2, y2, conf, cls)

        # 使用 label 数据（不可导），但不 detach detection box
        label_boxes = self.A_label[0].detach().cpu()
        loss_tfl = torch.tensor(0.0, device=self.device)

        # 遍历所有预测框
        for i in range(detections.size(0)):
            x1, y1, x2, y2 = detections[i]  
            d_box = torch.stack([x1, y1, x2, y2])  
            # 与每个真实框比较
            for label_box in label_boxes:
                _, x1_real, y1_real, x2_real, y2_real = label_box
                l_box = torch.tensor([x1_real, y1_real, x2_real, y2_real], device=self.device, dtype=d_box.dtype)
                iou = self.calculate_iou(d_box, l_box) 
                if iou > 0.5:
                    loss_tfl = loss_tfl + torch.nn.functional.mse_loss(d_box, l_box, reduction='sum')
        return loss_tfl
            # 我只能说很怪，这种用yolov3，loss怎么做反向传播，重新做一个直接预测坐标的yolov1网络可能还行，用yolov3主要是后处理NMS部分，没办法反向，只能重写NMS了
            # inv_normalize = transforms.Normalize(
            #     mean=[-1, -1, -1],
            #     std=[2, 2, 2]
            # )
            # img_0_1 = inv_normalize(img.detach())
            # img_0_1_np = (img_0_1.numpy()*255).astype(np.uint8)
            # img_0_1_np = img_0_1_np.transpose(1, 2, 0)
            # img_0_1_bgr = cv2.cvtColor(img_0_1_np, cv2.COLOR_RGB2BGR)
            # for label_box in label_boxes:
            #     _, x1_real, y1_real, x2_real, y2_real = label_box
            #     cv2.rectangle(img_0_1_bgr, (x1_real, y1_real), (x2_real, y2_real), (0, 0, 0), 2)
            #     print(x1_real, y1_real, x2_real, y2_real)

            # for box in detect_boxes:
            #     x1, y1, x2, y2, conf, cls = box
            #     cv2.rectangle(img_0_1_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.imshow("1",img_0_1_bgr)
            

            # img_s = inv_normalize(self.real_A[0].detach())
            # img_s_0_1 = (img_s.numpy()*255).astype(np.uint8)
            # img_s_0_1 = img_s_0_1.transpose(1, 2, 0)
            # img_s_0_1_bgr = cv2.cvtColor(img_s_0_1, cv2.COLOR_RGB2BGR)
            # cv2.imshow("2", img_s_0_1_bgr)

            # print(self.image_paths)
            # cv2.waitKey(0)
  
            # exit()
 
    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients

        # CycleGAN G Loss
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        if self.opt.CCD_GAN_stage == 2:
            # CCD-GAN stage2 G Loss
            # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
            pass
        else:
            # CCD-GAN stage1 G Loss
            self.loss_G = self.loss_G_A + self.loss_G_B

        self.loss_G.backward()

    def optimize_parameters(self):
        # """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # # forward
        # self.forward()      # compute fake images and reconstruction images.
        # # G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        # self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        # self.backward_G()             # calculate gradients for G_A and G_B
        # self.optimizer_G.step()       # update G_A and G_B's weights
        # # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        # self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        # self.backward_D_A()      # calculate gradients for D_A
        # self.backward_D_B()      # calculate graidents for D_B
        # self.optimizer_D.step()  # update D_A and D_B's weights
        if self.opt.CCD_GAN_stage == 2:
            self.optimize_stage_2()
        else:
            self.optimize_stage_1()

    def optimize_stage_1(self):
        # ==== Step 1: G_ST & D_T ====
        self.fake_B = self.netG_A(self.real_A)
        
        # train G_ST (netG_A)
        self.set_requires_grad(self.netD_A, False)
        self.optimizer_G.zero_grad()
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_A.backward(retain_graph=True)
        self.optimizer_G.step()

        # train D_T (netD_A)
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D.zero_grad()
        fake_B_pool = self.fake_B_pool.query(self.fake_B.detach())
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B_pool)
        self.optimizer_D.step()


        # ==== Step 2: G_TS & D_S ====
        self.fake_A = self.netG_B(self.real_B)  # fake_A = G_TS(real_B)

        # train G_TS (netG_B)
        self.set_requires_grad(self.netD_B, False)
        self.optimizer_G.zero_grad()
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_G_B.backward(retain_graph=True)
        self.optimizer_G.step()

        # train D_S (netD_B)
        self.set_requires_grad(self.netD_B, True)
        self.optimizer_D.zero_grad()
        fake_A_pool = self.fake_A_pool.query(self.fake_A.detach())
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A_pool)
        self.optimizer_D.step()

    def optimize_stage_2(self):
        torch.autograd.set_detect_anomaly(True)
        # ==== Step 1: G_ST & D_T ====
        self.fake_B = self.netG_A(self.real_A)  # fake_B = G_ST(real_A)
        
        # train G_ST (netG_A)
        self.set_requires_grad(self.netD_A, False)
        self.optimizer_G.zero_grad()
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_domain_A = self.domain_consistency_loss(self.fake_B, self.real_B)
        self.loss_content_A = self.content_consistency_loss(self.real_B, self.netG_B, self.netG_A)
        # equation 4,8,11
        self.loss_A = self.loss_G_A + self.lamda_D * self.loss_domain_A + self.lamda_C * self.loss_content_A
        self.loss_A.backward(retain_graph=True)
        self.optimizer_G.step()

        # train D_T (netD_A)
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D.zero_grad()
        fake_B_pool = self.fake_B_pool.query(self.fake_B.detach())
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B_pool)
        self.optimizer_D.step()


        # ==== Step 2: G_TS & D_S ====
        self.fake_A = self.netG_B(self.real_B)  # fake_A = G_TS(real_B)
        self.rec_A = self.netG_B(self.netG_A(self.real_A))  # rec_A = G_TS(G_ST(real_A))
        # train G_TS (netG_B)
        self.set_requires_grad(self.netD_B, False)
        self.optimizer_G.zero_grad()
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_domain_B = self.domain_consistency_loss(self.fake_A, self.real_A)
        self.loss_content_B = self.content_consistency_loss(self.real_A, self.netG_A, self.netG_B)

        self.loss_tfl = self.tfl_loss(self.rec_A)
        # equation 5,7,10,12
        self.loss_B = self.loss_G_B + self.lamda_D * self.loss_domain_B + self.lamda_C * self.loss_content_B + self.lamda_T * self.loss_tfl

        self.loss_B.backward(retain_graph=True)
        self.optimizer_G.step()

        # train D_S (netD_B)
        self.set_requires_grad(self.netD_B, True)
        self.optimizer_D.zero_grad()
        fake_A_pool = self.fake_A_pool.query(self.fake_A.detach())
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A_pool)
        self.optimizer_D.step()
