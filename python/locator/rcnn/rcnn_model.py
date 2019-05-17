import torch
import math
import torch.nn as nn
import misc.utils as utils
import misc.box_utils as box_utils
from misc.localization_layer import LocalizationLayer
from misc.resnet_blocks import BasicBlock, Bottleneck
from misc.apply_box_transform import ApplyBoxTransform
from misc.box_regression_criterion import BoxRegressionCriterion


class RcnnNet(nn.Module):
    def __init__(self, opt, logger):
        super(RcnnNet, self).__init__()
        utils.ensureopt(opt, 'mid_box_reg_weight')
        utils.ensureopt(opt, 'mid_objectness_weight')
        utils.ensureopt(opt, 'end_box_reg_weight')
        utils.ensureopt(opt, 'end_objectness_weight')
        utils.ensureopt(opt, 'box_reg_decay')

        self.opt = opt
        self.logger = logger
        x0, y0 = 0.0, 0.0
        sx, sy = 1.0, 1.0
        n = 4

        for i in range(n):
            x0 = x0 + sx / 2
            y0 = y0 + sy / 2
            sx = 2 * sx
            sy = 2 * sy

        # output from bilinear interpolation, ensures that the output from layer4 is 2 x 5
        # h x w
        output_size = (8, 20)
        # these parameters are also used in localization layer
        self.opt.output_size = output_size
        self.opt.input_dim = 128
        self.opt.cnn_dim = 512
        self.opt.field_centers = (x0, y0, sx, sy)

        # First part of resnet
        # BasicBlock a nn.Module, [3, 4, 6, 3]
        # note here basic block not initialization, only return class
        block, layers = self.get_block_and_layers(34)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3 basic blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 4 basic blocks
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.localization_layer = LocalizationLayer(self.opt, logger)
        # Rest of resnet
        # 6 basic blocks
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.avgpool = nn.AvgPool2d((2, 5))
        self.fc = nn.Linear(512 * block.expansion, 512 * block.expansion)

        # Initialize resnet weights
        self.init_resnet_weights()

        # Initialize localization_layer weights
        if opt.init_weights:
            self.localization_layer.init_weights()

        # Final box scoring layer
        self.box_scoring_branch = nn.Linear(512 * block.expansion, 2)

        # Final box regression layer
        self.apply_box_transform = ApplyBoxTransform()
        self.box_reg_branch = nn.Linear(512 * block.expansion, 4)
        if opt.init_weights:
            with torch.no_grad():
                self.box_scoring_branch.weight.normal_(0, self.opt.std)
                self.box_scoring_branch.bias.zero_()
                self.box_reg_branch.weight.zero_()
                self.box_reg_branch.bias.zero_()

        self.scoring_loss = nn.CrossEntropyLoss()
        self.box_reg_loss = BoxRegressionCriterion(self.opt.end_box_reg_weight)

    def load_weights(self, weight_file):
        if weight_file:
            self.load_state_dict(torch.load(weight_file))

    def init_resnet_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.zero_()

    def get_block_and_layers(self, num_layers):
        if num_layers == 34:
            out = BasicBlock, [3, 4, 6, 3]
        elif num_layers == 46:
            out = BasicBlock, [3, 4, 12, 3]
        elif num_layers == 50:
            out = Bottleneck, [3, 4, 6, 3]
        else:
            raise ValueError("invalid num_layer option")
        return out

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def setTestArgs(self, kwargs):
        self.localization_layer.setTestArgs({
            'nms_thresh': utils.getopt(kwargs, 'rpn_nms_thresh', 0.7),
            'max_proposals': utils.getopt(kwargs, 'max_proposals', 1000), })
        self.opt.final_nms_thresh = utils.getopt(kwargs, 'final_nms_thresh', 0.3)

    def setImageSize(self, image_height, image_width):
        self.localization_layer.setImageSize(image_height, image_width)

    def num_parameters(self):
        a = 0
        for p in self.parameters():
            a += p.data.nelement()
        return a

    def _eval_helper(self, image, boxes, final_adjust_boxes):
        """
        Feeds boxes through the network in batches so that we aren't limited
        by the GPU memory when it comes to number of boxes at test time.
        """
        scores, tboxes = [], []
        for v in boxes.split(self.opt.test_batch_size):
            with torch.no_grad():
                roi_feats = self.localization_layer.eval_boxes((image, v))
                roi_feats = self.layer3(roi_feats)
                roi_feats = self.layer4(roi_feats)
                roi_feats = self.bn2(roi_feats)
                roi_feats = self.relu(roi_feats)
                roi_feats = self.avgpool(roi_feats)
                roi_feats = roi_feats.view(roi_feats.size(0), -1)
                roi_codes = self.fc(roi_feats)
                s = self.box_scoring_branch(roi_codes).cpu()
                if final_adjust_boxes:
                    box_trans = self.box_reg_branch(roi_codes)
                    b = self.apply_box_transform((v, box_trans.data)).cpu()
                    tboxes.append(b)

                scores.append(s.data)

        scores = torch.cat(scores, dim=0)
        out = (scores,)
        if final_adjust_boxes:
            tboxes = torch.cat(tboxes, dim=0)
            out += (tboxes,)

        return out

    def evaluate(self, input, args, cpu=True):
        image, gt_boxes = input[0], input[1]
        if not torch.cuda.is_available():
            self.logger.warning('Could not find CUDA environment, using CPU mode')
            args.gpu = False
        device = torch.device("cuda:0" if args.gpu else "cpu")
        image = image.to(device)
        if args.use_external_proposals:
            proposals = input[2]
            proposals = proposals.to(device)
        B, C, H, W = image.shape
        self.setImageSize(H, W)
        with torch.no_grad():
            image = self.conv1(image)
            image = self.bn1(image)
            image = self.relu(image)
            image = self.maxpool(image)
            image = self.layer1(image)
            image = self.layer2(image)
            # do nms for roi_boxes
            roi_boxes = self.localization_layer(image)
            roi_scores, roi_boxes = self._eval_helper(image, roi_boxes, True)
            proposal_scores = self._eval_helper(image, proposals, False)[0]

            # Convert to x1y1x2y2
            roi_boxes = box_utils.xcycwh_to_x1y1x2y2(roi_boxes)

            if cpu:
                roi_scores = roi_scores.cpu()
                roi_boxes = roi_boxes.cpu()
                if args.use_external_proposals:
                    proposal_scores = proposal_scores.cpu()

            if args.numpy:
                # Convert to numpy array
                roi_scores = roi_scores.cpu().numpy()
                roi_boxes = roi_boxes.cpu().numpy()
                if args.use_external_proposals:
                    proposal_scores = proposal_scores.cpu().numpy()

            roi_scores = roi_scores[:, 1]

            out = (roi_scores, roi_boxes)
            if args.use_external_proposals:
                proposal_scores = proposal_scores[:, 1]
                out += (proposal_scores,)
        return out

    def forward(self, input):
        image, gt_boxes = input[0], input[1]
        # set localizatin layer image size
        self.setImageSize(image.size(2), image.size(3))
        image = self.conv1(image)
        image = self.bn1(image)
        image = self.relu(image)
        image = self.maxpool(image)
        image = self.layer1(image)
        image = self.layer2(image)
        # 1, 128, 215, 133
        ll_in = (image, gt_boxes)
        # fix this after remove label from dataset
        if self.opt.dtp_train:
            ll_in += (input[2],)

        if self.opt.dtp_train:
            roi_feats, roi_boxes, pos_target_boxes, mid_loss, rpn_pos = self.localization_layer(ll_in)
        else:
            roi_feats, roi_boxes, pos_target_boxes, mid_loss = self.localization_layer(ll_in)
        roi_feats = self.layer3(roi_feats)

        roi_feats = self.layer4(roi_feats)
        roi_feats = self.bn2(roi_feats)
        roi_feats = self.relu(roi_feats)
        roi_feats = self.avgpool(roi_feats)
        roi_feats = roi_feats.view(roi_feats.size(0), -1)
        roi_codes = self.fc(roi_feats)
        num_pos = pos_target_boxes.size(0)
        # scores are based on roi_codes
        scores = self.box_scoring_branch(roi_codes)
        pos_roi_codes = roi_codes[:num_pos]
        pos_roi_boxes = roi_boxes[:num_pos]

        # Don't do box regression on roi boxes since we don't try to adjust dtp boxes
        if self.opt.dtp_train:
            reg_roi_codes = pos_roi_codes[:rpn_pos]
            reg_roi_boxes = pos_roi_boxes[:rpn_pos]
            box_trans = self.box_reg_branch(reg_roi_codes)
            boxes = self.apply_box_transform((reg_roi_boxes, box_trans))
            out = (scores, pos_roi_boxes, box_trans, boxes, pos_target_boxes,
                   mid_loss, rpn_pos)
        else:
            # reg are based on pos_roi_codes
            box_trans = self.box_reg_branch(pos_roi_codes)
            # do transformation again for positive boxes
            boxes = self.apply_box_transform((pos_roi_boxes, box_trans))
            out = (scores, pos_roi_boxes, box_trans, boxes, pos_target_boxes,
                   mid_loss)
        return self.post_processing(out)

    def post_processing(self, out):
        wordness_scores = out[0]
        pos_roi_boxes = out[1]
        predict_boxes = out[3]
        final_box_trans = out[2]
        gt_boxes = out[4]
        mid_loss = out[5]

        num_boxes = wordness_scores.size(0)
        num_pos = pos_roi_boxes.size(0)
        # Compute final objectness loss and gradient
        wordness_labels = torch.zeros(num_boxes, dtype=torch.long).view(-1, 1)
        wordness_labels[:num_pos].fill_(1)
        if self.opt.gpu:
            wordness_labels = wordness_labels.cuda()

        wordness_labels = wordness_labels.view(-1)

        end_objectness_loss = self.scoring_loss.forward(wordness_scores, wordness_labels) \
                              * self.opt.end_objectness_weight

        if self.opt.dtp_train:
            rpn_pos = out[6]
            pos_roi_boxes = pos_roi_boxes[:rpn_pos]
            gt_boxes = gt_boxes[:rpn_pos]

        # this one multiplies by the weight inside the loss so we don't do it manually.
        end_box_reg_loss = self.box_reg_loss.forward((pos_roi_boxes, final_box_trans), gt_boxes)

        total_loss = mid_loss + end_objectness_loss + end_box_reg_loss
        total_loss.backward()
        ll_losses = self.localization_layer.stats
        losses = {
            'mid-obj': ll_losses.obj_loss_pos.cpu() + ll_losses.obj_loss_neg.cpu(),
            'box-decay': ll_losses.box_decay_loss.cpu(),
            'mid-reg': ll_losses.box_reg_loss.cpu(),
            'end-obj': end_objectness_loss.detach().cpu(),
            'end-reg': end_box_reg_loss.detach().cpu(),
            'total_loss': total_loss.detach().cpu()}

        for k, v in losses.items():
            losses[k] = v.item()

        return losses, predict_boxes
