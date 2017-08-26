require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
if opt.cuda then
    require 'cunn'
end

----------------------------------------------------------------------
print '==> defining validation procedure'

validationLogger = optim.Logger('validation.log')

-- validation function
function validate()
    -- local vars
    local time = sys.clock()
    --[[
        -- averaged param use?
        if average then
            cachedparams = parameters:clone()
            parameters:copy(average)
        end]]

    -- Global error
    local global_err = 0

    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model:evaluate()

    -- test over test data
    print('=================================')
    print('==> validating on validation set:')
    local batch_count = 0
    for t = 1,validationData:size(),batchSize do
        -- disp progress
        xlua.progress(t, validationData:size())
        batch_count = batch_count + 1

        local currentBatchSize = math.min(batchSize, trainData:size() - t + 1)

        -- create mini batch
        local inputs = torch.Tensor(currentBatchSize, input_count)
        local targets = torch.Tensor(currentBatchSize, output_channel_count, ouput_dim, ouput_dim)
        local k = 1
        for i = t,math.min(t+batchSize-1, validationData:size()) do
            -- load new sample
            local input = validationData.data[i]
            local target = validationData.target[i]
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end

        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        -- evaluate function for complete mini batch
        local preds = model:forward(inputs)
        local f = criterion:forward(preds, targets)
        --print(preds)

        -- Display first image of batch
        if opt.visualization then
            local batch_targets_image = expandBatchToSpatial(targets:float())
            local batch_preds_image = expandBatchToSpatial(preds:float())
            w_val_target = image.display{image=batch_targets_image, offscreen=false, win=w_val_target}
            w_val_pred = image.display{image=batch_preds_image, offscreen=false, win=w_val_pred}
            if opt.save_visualization then
                image.save("validation_batch_groundtruths.png", batch_targets_image)
                image.save("validation_batch_predictions.png", batch_preds_image)
            end
        end

        -- update global error
        global_err = global_err + f

    end

    -- timing
    time = sys.clock() - time
    time = time / validationData:size()
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

    -- Global error
    global_err = global_err / batch_count
    print("global validation err = " .. global_err)

    -- update logger/plot
    validationLogger:add{['% mean err (validation set)'] = global_err}
    validationLogger:style{['% mean err (validation set)'] = '-'}
    --validationLogger:plot()

    --[[
    -- averaged param use?
    if average then
        -- restore parameters
        parameters:copy(cachedparams)
    end
    ]]

    -- next iteration:
end