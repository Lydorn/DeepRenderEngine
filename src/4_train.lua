require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
if opt.cuda then
    require 'cunn'
end


local function deepCopy(tbl)
    -- creates a copy of a network with new modules and the same tensors
    local copy = {}
    for k, v in pairs(tbl) do
        if type(v) == 'table' then
            copy[k] = deepCopy(v)
        else
            copy[k] = v
        end
    end
    if torch.typename(tbl) then
        torch.setmetatable(copy, torch.typename(tbl))
    end
    return copy
end

-- Params
learningRate = 1e-4
learningRateDecay = 1e-3
weightDecay = 0
batchSize = math.min(256, trainData:size(), validationData:size())

if opt.cuda then
    model:cuda()
    criterion:cuda()
end

--------------
-- Log results to files
trainLogger = optim.Logger('train.log')

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the model
-- into a 1-dim vector
if model then
    parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring adam optimizer'

optimState = {
    learningRate = learningRate,
    weightDecay = weightDecay,
    learningRateDecay = learningRateDecay
}
optimMethod = optim.adam

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- Global error
    local global_err = 0

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- shuffle at each epoch
    shuffle = torch.randperm(train_size)

    -- do one epoch
    print('=================================')
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
    local batch_count = 0
    for t = 1,trainData:size(),batchSize do
        -- disp progress
        xlua.progress(t, trainData:size())
        batch_count = batch_count + 1

        local currentBatchSize = math.min(batchSize, trainData:size() - t + 1)

        -- create mini batch
        local inputs = torch.Tensor(currentBatchSize, input_count)
        local targets = torch.Tensor(currentBatchSize, output_channel_count, ouput_dim, ouput_dim)
        local k = 1
        for i = t,math.min(t+batchSize-1, trainData:size()) do
            -- load new sample
            local input = trainData.data[shuffle[i]]
            local target = trainData.target[shuffle[i]]
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end

        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)

            -- Display first image of batch
            w_train_target = image.display{image=targets[1], offscreen=false, win=w_train_target}
            w_train_output = image.display{image=outputs[1], offscreen=false, win=w_train_output}


            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            -- update global error
            global_err = global_err + f

            -- return f and df/dX
            return f,gradParameters
        end

        -- optimize on current mini-batch
        optimMethod(feval, parameters, optimState)
    end

    -- time taken
    time = sys.clock() - time
    time = time / trainData:size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- Global error
    global_err = global_err / batch_count
    print("global train err = " .. global_err)

    -- update logger/plot
    trainLogger:add{['% mean err (train set)'] = global_err}
    trainLogger:style{['% mean err (train set)'] = '-'}
    --trainLogger:plot()

    -- Vizualize weights
    --win_weights = image.display{image=model:get(1).weight, offscreen=false, win=win_weights}
    --win_outputs = image.display{image=model:get(1).output, offscreen=false, win=win_outputs}

    -- save/log current net
    --print('==> saving model to '.. opt.model_filename)
    local model_save= deepCopy(model):float():clearState()
    torch.save(opt.model_filename, model_save)

    -- next epoch
    epoch = epoch + 1
end
