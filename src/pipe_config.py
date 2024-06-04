def splite_model(pipe, pipe_id, n):
    unet = pipe.unet
    if pipe_id == "stabilityai/stable-video-diffusion-img2vid-xt":
        if n == 2:
            return [
                (unet.conv_in, *unet.down_blocks, unet.mid_block, unet.up_blocks[0], unet.up_blocks[1]),
                (*unet.up_blocks[2:], unet.conv_norm_out, unet.conv_out),
            ]
        elif n == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif pipe_id == "stabilityai/stable-diffusion-2-1":
        if n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                *unet.up_blocks[:1],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[1],
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[2],
            ), (
                unet.up_blocks[1].attentions[2],
                *unet.up_blocks[1].upsamplers,
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            ),]
    else:
        raise NotImplementedError
    
