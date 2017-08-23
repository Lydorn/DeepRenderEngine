require 'torch'
require 'image'

--Params
opt = {
    model_filename = 'work/model.net',
    cuda = true
}

torch.setdefaulttensortype('torch.FloatTensor')

torch.manualSeed(2)

----------------------------------------------------------------------
print '==> executing all'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_validate.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
    train()
    validate()
end
