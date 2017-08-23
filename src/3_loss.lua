-- TODO: implement smarter loss (using HSV maybe?)

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions


criterion = nn.MSECriterion()

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)
