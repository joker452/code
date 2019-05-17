"""
Created on Thu Oct 12 23:01:57 2017

@author: tomas
"""
import torch
from . import utils
from . import box_sampler

class BoxSamplerHelper(torch.nn.Module):
    def __init__(self, opt, logger):
        super(BoxSamplerHelper, self).__init__()
        self.box_sampler = box_sampler.BoxSampler(opt, logger)

    def setBounds(self, bounds):
        self.box_sampler.setBounds(bounds)

    """
      Input:
      
      List of two lists. The first list contains data about the input boxes,
      and the second list contains data about the target boxes.

      The first element of the first list is input_boxes, a Tensor of shape (N, B1, 4)
      giving coordinates of the input boxes in (xc, yc, w, h) format.

      All other elements of the first list are tensors of shape (N, B1, Di) parallel to
      input_boxes; Di can be different for each element.

      The first element of the second list is target_boxes, a Tensor of shape (N, B2, 4)
      giving coordinates of the target boxes in (xc, yc, w, h) format.

      All other elements of the second list are tensors of shape (N, B2, Dj) parallel
      to target_boxes; Dj can be different for each Tensor.

      
      Returns a list of three lists:

      The first list contains data about positive input boxes. The first element is of
      shape (P, 4) and contains coordinates of positive boxes; the other elements
      correspond to the additional input data about the input boxes; in particular the
      ith element has shape (P, Di).

      The second list contains data about target boxes corresponding to positive
      input boxes. The first element is of shape (P, 4) and contains coordinates of
      target boxes corresponding to sampled positive input boxes; the other elements
      correspond to the additional input data about the target boxes; in particular the
      jth element has shape (P, Dj).

      The third list contains data about negative input boxes. The first element is of
      shape (M, 4) and contains coordinates of negative input boxes; the other elements
      correspond to the additional input data about the input boxes; in particular the
      ith element has shape (M, Di).
    """

    def forward(self, input):
        input_data = input[0]
        target_data = input[1]
        input_boxes = input_data[0]
        target_boxes = target_data[0]
        N = input_boxes.size(0)
        assert N == 1, 'Only minibatches of 1 are supported'

        # Run the sampler to get the indices of the positive and negative boxes
        idxs = self.box_sampler.forward((input_boxes, target_boxes))
        pos_input_idx = idxs[0]
        pos_target_idx = idxs[1]
        neg_input_idx = idxs[2]

        # Resize the output. We need to allocate additional tensors for the
        # input data and target data, then resize them to the right size.
        num_pos = pos_input_idx.size(0)
        num_neg = neg_input_idx.size(0)

        # -- Now use the indeces to actually copy data from inputs to outputs
        pos, target, neg = [], [], []
        for i in range(len(input_data)):
            d = input_data[i]
            D = d.size(2)
            pos.append(d[:, pos_input_idx].view(num_pos, D))
            if i == 0 or i == 3:
                neg.append(d[:, neg_input_idx].view(num_neg, D))

        for i in range(len(target_data)):
            d = target_data[i]
            D = d.size(2)

            target.append(d[:, pos_target_idx].view(num_pos, D))

        output = (pos, target, neg)

        return output
