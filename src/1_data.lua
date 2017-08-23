--[[
-- Possible data augmentation: swap Radian and Dire OR share/reflect params in some way (better)
---------------------------------------------------------------------
print '==> Augmenting data'
]]

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

-- Params
data_size  = 1000
train_size =  900
validation_size = data_size - train_size
input_count = 6 -- Sun angles X and Z, Sun color RGB and strength

output_channel_count = 3
ouput_dim = 32

data_train_path = "work/data_train.t7"
data_validation_path = "work/data_validation.t7"

gt_img_path_format     = "../data/groundtruth_images/%.6d.png"
scene_data_path = "../data/scene_data.csv"

if (file_exists(data_train_path) and file_exists(data_validation_path)) then
    print '==> Loading existing datasets...'
    trainData = torch.load(data_train_path, 'ascii')
    validationData = torch.load(data_validation_path, 'ascii')
else
    local csv = require("csv")
    -------------------------------------------------------------------
    scene_data = {};
    gt_img = {};

    X = torch.Tensor(data_size, input_count):zero()
    Y = torch.Tensor(data_size, output_channel_count, ouput_dim, ouput_dim):zero()

    -------------------------------------------------------------------
    print('==> Loading scene data...')

    local scene_data_file = csv.open(scene_data_path)
    local line_index = 1
    for fields in scene_data_file:lines() do
        xlua.progress(line_index, data_size)
        if (1 < line_index) then
            local scene_data_tensor = torch.Tensor(input_count)
            for input_i = 1, input_count do
                scene_data_tensor[input_i] = fields[input_i]
            end
            table.insert(scene_data, scene_data_tensor)
        end
        line_index = line_index + 1
    end

    print('==> Loading groundtruth images...')
    for i = 1, data_size do
        xlua.progress(i, data_size)
        local img = image.load(string.format(gt_img_path_format, i - 1), output_channel_count)
				w_gt_img = image.display{image=img, offscreen=false, win=w_gt_img}
        table.insert(gt_img, img);
        i = i + 1
    end

    -------------------------------------------------------------------
    print('==> Building dataset for Torch...')

    for scene_index = 1,data_size do
        X[scene_index] = scene_data[scene_index]
        Y[scene_index] = gt_img[scene_index]
    end

    -------------------------------------------------------------------
    print('==>  Splitting into train and validation sets...')

    shuffle = torch.randperm(data_size)

    trainData = {
        data = X:index(1,shuffle:index(1, torch.range(1, train_size):long()):long()),
        target = Y:index(1,shuffle:index(1, torch.range(1, train_size):long()):long()),
        size = function() return train_size end
    }

    validationData = {
        data = X:index(1,shuffle:index(1, torch.range(train_size + 1, data_size):long()):long()),
        target = Y:index(1,shuffle:index(1, torch.range(train_size + 1, data_size):long()):long()),
        size = function() return validation_size end
    }

    -------------------------------------------------------------------
    print("===> Saving data...")
    torch.save(data_train_path, trainData, 'ascii')
    torch.save(data_validation_path, validationData, 'ascii')
end
