def splite_model(pipe, pipe_id, n):
    unet = pipe.unet
    if pipe_id == "stabilityai/stable-video-diffusion-img2vid-xt":
        if n == 2:
            return [
                (
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ),
                (
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )
            ]
        elif n == 3:
            return [
                (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.down_blocks[2],
            ),
                (
                unet.down_blocks[3],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
            ),
                (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )
            ]
        elif n == 4:
            return [
                (
                unet.down_blocks[1].resnets[0],
                unet.down_blocks[1].attentions[0],
                unet.conv_in,
                unet.down_blocks[0],
            ),
                (
                unet.down_blocks[1].resnets[1],
                unet.down_blocks[1].attentions[1],
                *unet.down_blocks[1].downsamplers,
                *unet.down_blocks[2:4],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ),
                (
                unet.up_blocks[2],
                unet.up_blocks[3].resnets[0],
                unet.up_blocks[3].attentions[0],
            ),
            (
                unet.up_blocks[3].resnets[1],
                unet.up_blocks[3].attentions[1],
                unet.up_blocks[3].resnets[2],
                unet.up_blocks[3].attentions[2],
                unet.conv_norm_out,
                unet.conv_out
            )
            ]
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
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.down_blocks[2],
                unet.down_blocks[3],
            ), (
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
            ), (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 4:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1]
            ), (
                *unet.down_blocks[2:4],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ), (
                unet.up_blocks[2],
                unet.up_blocks[3].resnets[0],
            ), (
                unet.up_blocks[3].attentions[0],
                unet.up_blocks[3].resnets[1],
                unet.up_blocks[3].attentions[1],
                unet.up_blocks[3].resnets[2],
                unet.up_blocks[3].attentions[2],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "runwayml/stable-diffusion-v1-5": 
        if n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                *unet.up_blocks[:1],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[1],
            ), (
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[2],
                unet.up_blocks[1].attentions[2],
                *unet.up_blocks[1].upsamplers,
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.down_blocks[2],
                unet.down_blocks[3],
            ), (
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2].resnets[0],
                unet.up_blocks[2].attentions[0],
                unet.up_blocks[2].resnets[1],
                unet.up_blocks[2].attentions[1],
                unet.up_blocks[2].resnets[2],
            ), (
                unet.up_blocks[2].attentions[2],
                *unet.up_blocks[2].upsamplers,
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 4:
            return [(
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1]
            ), (
                *unet.down_blocks[2:4],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[1],
            ), (
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[2],
                unet.up_blocks[1].attentions[2],
                *unet.up_blocks[1].upsamplers,
                unet.up_blocks[2],
            ), (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "stabilityai/stable-diffusion-xl-base-1.0" or pipe_id == "RunDiffusion/Juggernaut-X-v10": 
        if n == 2:
            return [(
                unet.down_blocks[2],
                unet.mid_block,
                unet.up_blocks[0].resnets[0],
                unet.up_blocks[0].attentions[0],
                unet.up_blocks[0].resnets[1],
                unet.up_blocks[0].attentions[1],
            ), (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.conv_norm_out,
                unet.conv_out,
                unet.up_blocks[0].resnets[2],
                unet.up_blocks[0].attentions[2],
                *unet.up_blocks[0].upsamplers,
            )]
        elif n == 3:
            return [(
                unet.down_blocks[2],
                unet.mid_block,
                unet.up_blocks[0].resnets[0],
            ), (
                unet.up_blocks[0].attentions[0],
                unet.up_blocks[0].resnets[1],
                unet.up_blocks[0].attentions[1],
                unet.up_blocks[0].resnets[2],
                unet.up_blocks[0].attentions[2],
                *unet.up_blocks[0].upsamplers,
            ), (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.conv_norm_out,
                unet.conv_out,
            )]
        elif n == 4:
            return [(
                unet.down_blocks[1].attentions[0],
                unet.down_blocks[1].resnets[1],
                unet.down_blocks[1].attentions[1],
                *unet.down_blocks[1].downsamplers,
                unet.down_blocks[2]
            ), (
                unet.mid_block,
                unet.up_blocks[0].resnets[0],
                unet.up_blocks[0].attentions[0],
            ), (
                unet.up_blocks[0].resnets[1],
                unet.up_blocks[0].attentions[1],
                unet.up_blocks[0].resnets[2],
                unet.up_blocks[0].attentions[2],
                *unet.up_blocks[0].upsamplers,
            ), (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1].resnets[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.conv_norm_out,
                unet.conv_out,
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "emilianJR/epiCRealism": 
        if n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ), (
                unet.up_blocks[2],
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                *unet.down_blocks,
            ), (
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
            ), (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    elif pipe_id == "stabilityai/stable-diffusion-x4-upscaler": 
        if n == 2:
            return [(
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1].attentions[0],
                unet.up_blocks[1].resnets[0],
                unet.up_blocks[1].attentions[1],
                unet.up_blocks[1].resnets[1],
            ), (
                unet.up_blocks[1].attentions[2],
                unet.up_blocks[1].resnets[2],
                *unet.up_blocks[0].upsamplers,
                unet.up_blocks[2],
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 3:
            return [(
                unet.conv_in,
                *unet.down_blocks,
            ), (
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2].attentions[0],
                unet.up_blocks[2].resnets[0],
                unet.up_blocks[2].attentions[1],
            ), (
                unet.up_blocks[2].resnets[1],
                unet.up_blocks[2].attentions[2],
                unet.up_blocks[2].resnets[2],
                *unet.up_blocks[2].upsamplers,
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        elif n == 4:
            return [(
                unet.conv_in,
                *unet.down_blocks[0:3],
            ), (
                unet.down_blocks[3],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            ),
                (
                unet.up_blocks[2]   
            ),
                (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
