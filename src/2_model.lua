require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

if opt.cuda then
    require 'cunn'
end

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

----------------------------------------------------------------------

if (file_exists(opt.model_filename)) then
    print '==> Loading existing model...'
    model = torch.load(opt.model_filename)
else
    ----------------------------------------------------------------------
    print '==> Defining new model...'

    ----------------------------------------------------------------------
    print '==> construct model'

    model = nn.Sequential()
    model:add(nn.Linear(input_count, 256*8*8))
    model:add(nn.Reshape(256, 8, 8))
    model:add(nn.ReLU(true))
    -- spatial dim = 8
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(nn.SpatialConvolution(256, 512, 5, 5, 1, 1, 2, 2))
    model:add(nn.SpatialBatchNormalization(512))
    model:add(nn.ReLU(true))
    -- spatial dim = 16
    model:add(nn.SpatialUpSamplingNearest(2))
    model:add(nn.SpatialConvolution(512, 256, 5, 5, 1, 1, 2, 2))
    model:add(nn.SpatialBatchNormalization(256))
    model:add(nn.ReLU(true))
    -- spatial dim = 32
    model:add(nn.SpatialConvolution(256, output_channel_count, 3, 3, 1, 1, 1, 1))

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization

--[[
generateGraph = require 'optnet.graphgen'

graphOpts = {
    displayProps =  {shape='ellipse',fontsize=14, style='solid'},
    nodeData = function(oldData, tensor)
        return oldData .. '\n' .. 'Size: '.. tensor:numel()
    end
}

input = torch.FloatTensor(team_count, player_count, hero_count)

g = generateGraph(model, input, graphOpts)

graph.dot(g, opt.model_filename, opt.model_filename)
]]
