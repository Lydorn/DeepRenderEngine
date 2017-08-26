function expandBatchToSpatial(batch_image_tensor)
    local batch_size = (#batch_image_tensor)[1]
    local channel_count = (#batch_image_tensor)[2]
    local spatial_res = (#batch_image_tensor)[3]

    local new_spatial_res_in_images = math.sqrt(batch_size)
    local new_spatial_res = math.ceil(new_spatial_res_in_images * spatial_res)
    local new_image_tensor = torch.Tensor(channel_count, new_spatial_res, new_spatial_res):zero()

    for batch_index = 0,(batch_size - 1) do
        local i = math.floor(batch_index / new_spatial_res_in_images)
        local j = batch_index % new_spatial_res_in_images
        new_image_tensor[{{}, {i * spatial_res + 1, (i + 1) * spatial_res}, {j * spatial_res + 1, (j + 1) * spatial_res}}] = batch_image_tensor[batch_index + 1]
    end

    return new_image_tensor
end