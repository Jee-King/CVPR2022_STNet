from typing import Dict

from videoanalyst.data.utils.crop_track_pair import crop_track_pair

from ..transformer_base import TRACK_TRANSFORMERS, TransformerBase


@TRACK_TRANSFORMERS.register
class RandomCropTransformer(TransformerBase):
    r"""
    Cropping training pair with data augmentation (random shift / random scaling)

    Hyper-parameters
    ----------------

    context_amount: float
        the context factor for template image
    max_scale: float
        the max scale change ratio for search image
    max_shift:  float
        the max shift change ratio for search image
    max_scale_temp: float
        the max scale change ratio for template image
    max_shift_temp:  float
        the max shift change ratio for template image
    z_size: int
        output size of template image
    x_size: int
        output size of search image
    """
    default_hyper_params = dict(
        context_amount=0.5,
        max_scale=0.3,
        max_shift=0.4,
        max_scale_temp=0.0,
        max_shift_temp=0.0,
        z_size=127,
        x_size=303,
    )

    def __init__(self, seed: int = 0) -> None:
        super(RandomCropTransformer, self).__init__(seed=seed)

    def __call__(self, sampled_data: Dict) -> Dict:
        r"""
        sampled_data: Dict()
            input data
            Dict(data1=Dict(image, anno), data2=Dict(image, anno))
        """
        data1_pos = sampled_data["data1_pos"]
        data1_neg = sampled_data["data1_neg"]
        data2_pos = sampled_data["data2_pos"]
        data2_neg = sampled_data["data2_neg"]
        im_temp_pos, bbox_temp = data1_pos["image"], data1_pos["anno"]
        im_temp_neg = data1_neg['image']
        im_curr_pos, bbox_curr = data2_pos["image"], data2_pos["anno"]
        im_curr_neg = data2_neg['image']
        im_z_pos, im_z_neg, bbox_z, im_x_pos, im_x_neg, bbox_x, _, _ = crop_track_pair(
            im_temp_pos,
            im_temp_neg,
            bbox_temp,
            im_curr_pos,
            im_curr_neg,
            bbox_curr,
            config=self._hyper_params,
            rng=self._state["rng"])

        sampled_data["data1_pos"] = dict(image=im_z_pos, anno=bbox_z)
        sampled_data["data1_neg"] = dict(image=im_z_neg, anno=bbox_z)
        sampled_data["data2_pos"] = dict(image=im_x_pos, anno=bbox_x)
        sampled_data["data2_neg"] = dict(image=im_x_neg, anno=bbox_x)

        return sampled_data
