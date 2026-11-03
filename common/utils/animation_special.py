"""Reviewed garment and vehicle parts missed by the human semantic classes."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from utils.animation_parts import Layer, _complete_rgb, _cut, composite
from utils.animation_integrity import _owner_map, polygon


def preserve_visible_source(src: np.ndarray, layers: list[Layer], target_mask: np.ndarray | None = None) -> list[Layer]:
    """Assign missing edge pixels to named parts; keep the observed rest pose."""
    target = src[...,3]>10
    if target_mask is not None:
        target &= target_mask
    layers = [p for p in layers if p.mask.any()]
    labels = np.zeros(target.shape,np.int16)
    for index,p in enumerate(layers,1):
        p.rgba[~target] = 0
        labels[p.mask] = index
    missing = target & (labels==0)
    if missing.any():
        nearest = ndi.distance_transform_edt(labels==0,return_distances=False,return_indices=True)
        assigned = labels[tuple(nearest)]
        for index,p in enumerate(layers,1):
            mask = missing & (assigned==index)
            p.rgba[mask] = src[mask]
    owner = _owner_map(layers,target)
    if (owner[target]<0).any():
        raise ValueError('Reviewed parts do not cover the source foreground')
    edge = target & (src[...,3]<255)
    for index,p in enumerate(layers):
        selected = owner==index
        p.rgba[selected] = src[selected]
        p.rgba[edge & ~selected,3] = 0
    rendered = composite(layers)
    expected = src.copy();expected[~target]=0
    if not np.array_equal(rendered[...,3],expected[...,3]):
        raise ValueError('Reviewed source alpha changed')
    if target.any() and np.abs(rendered[target,:3].astype(int)-src[target,:3].astype(int)).max()>1:
        raise ValueError('Reviewed source colours changed')
    return [p for p in layers if p.mask.any()]


def _rounded_box(shape: tuple[int,int], box: tuple[int,int,int,int], radius: int) -> np.ndarray:
    x1,y1,x2,y2 = box
    mask = np.zeros(shape,np.uint8)
    cv2.rectangle(mask,(x1+radius,y1),(x2-radius,y2),1,-1)
    cv2.rectangle(mask,(x1,y1+radius),(x2,y2-radius),1,-1)
    for x in (x1+radius,x2-radius):
        for y in (y1+radius,y2-radius):
            cv2.circle(mask,(x,y),radius,1,-1)
    return mask>0


def _lamp_mask(src: np.ndarray, centre: tuple[int,int]) -> np.ndarray:
    yy,xx = np.indices(src.shape[:2]);cx,cy=centre
    nearby = (xx-cx)**2+(yy-cy)**2<70**2
    coloured = (src[...,1]<150) | (src[...,0]<110)
    raw = (nearby & coloured & (src[...,3]>10)).astype(np.uint8)
    _,labels,_,_ = cv2.connectedComponentsWithStats(raw,8)
    label = labels[cy,cx]
    if label==0:
        raise ValueError('Lamp centre is not inside its reviewed colour component')
    return ndi.binary_fill_holes(labels==label)


def split_rear_car(src: np.ndarray, use_lama: bool = True) -> list[Layer]:
    """Rear-view tail lamps, mirrors and two partially occluded tyres."""
    target = src[...,3]>10
    shape=target.shape;yy,xx=np.indices(shape)
    lamps=[_lamp_mask(src,(149,554)),_lamp_mask(src,(873,554))]
    left_points=[[21,381],[100,381],[112,389],[118,411],[125,424],[136,431],
                 [132,447],[111,458],[21,461],[7,451],[3,436],[3,405],[8,389]]
    mirrors=[polygon(shape,left_points)&target,
             polygon(shape,[[1023-x,y] for x,y in left_points])&target]
    tyres=[target & (yy>=807) & (xx>=48) & (xx<=214),
           target & (yy>=807) & (xx>=809) & (xx<=975)]
    mechanical=np.logical_or.reduce(mirrors+tyres)
    body_support=target & ~mechanical
    lamp_union=lamps[0]|lamps[1]
    body_rgba,completed=_complete_rgb(src,body_support & ~lamp_union,body_support,use_lama)
    body=Layer('car_body',body_rgba,np.full(shape,150,np.uint8),'vehicle',role='vehicle_body',
               meta={'label':'车身','provenance':'source_visible_vehicle_parts',
                     'inferred_under_lamps':int(completed.sum())})
    parts=[]
    for index,side in enumerate(('left','right')):
        rgba=src.copy();rgba[~lamps[index]]=0
        parts.append(Layer(f'tail_light_{side}',rgba,np.full(shape,65,np.uint8),'vehicle',role='vehicle_light',
                           parent='car_body',pivot=[149 if index==0 else 873,554],
                           meta={'label':f'画面{"左" if index==0 else "右"}尾灯','view':'rear'}))
        rgba=src.copy();rgba[~mirrors[index]]=0
        parts.append(Layer(f'mirror_{side}',rgba,np.full(shape,70,np.uint8),'vehicle',role='vehicle_mirror',
                           parent='car_body',pivot=[134 if index==0 else 889,435],
                           meta={'label':f'画面{"左" if index==0 else "右"}后视镜'}))
        box=(53,680,212,851) if index==0 else (811,680,974,851)
        support=_rounded_box(shape,box,24)&target
        support|=tyres[index]
        # The visible strip includes the red bumper edge and a black watermark.
        # Only clean rubber may seed the hidden tyre; observed pixels are restored
        # by preserve_visible_source below, including the original watermark.
        rgb=src[...,:3].astype(np.int16)
        rubber=tyres[index] & (rgb.max(axis=-1)<90) & (rgb.sum(axis=-1)>35) & (rgb[...,2]>rgb[...,0]+3)
        rubber=cv2.erode(rubber.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        if rubber.sum()<24:
            raise ValueError('No reliable visible rubber sample for the reviewed tyre')
        tyre_rgba,completed=_complete_rgb(src,rubber,support,False)
        parts.append(Layer(f'tyre_{side}',tyre_rgba,np.full(shape,220,np.uint8),'vehicle',role='vehicle_tyre',
                           parent='car_body',pivot=[132 if index==0 else 893,766],
                           meta={'label':f'画面{"左" if index==0 else "右"}轮胎',
                                 'hidden_outline':'bounded_estimate_behind_body',
                                 'completion':'conservative_visible_rubber_continuation',
                                 'inferred_hidden_pixels':int(completed.sum())}))
    return preserve_visible_source(src,[*parts,body])


def split_knight_skirts(src: np.ndarray, layers: list[Layer], use_lama: bool = True) -> list[Layer]:
    """Front apron, side cloth and real dark lining behind the leg armour."""
    top=next(p for p in layers if p.name=='topwear')
    head=next(p for p in layers if p.name=='headwear')
    shape=top.mask.shape;yy,xx=np.indices(shape)
    target=src[...,3]>10
    limb=[p for p in layers if p.role.startswith(('arm_','leg_'))]
    for p in limb:
        p.depth[:]=60 if p.role.startswith('arm_') else 150
        if p.role in {'arm_upper','leg_thigh','arm_whole','leg_whole'}:
            p.parent=f'{p.name.split("_")[0]}_torso'
    result=list(limb)
    centre=[[240,478],[301,461],[344,480],[349,745],[333,784],[301,820],[263,785],[243,747]]
    hip_left=[[183,433],[222,441],[246,469],[247,515],[214,542],[171,562],
              [126,582],[141,539],[158,497],[179,473]]
    hip_right=[[343,434],[364,439],[370,470],[390,512],[417,579],[375,559],[348,540],[335,513],[339,469]]
    side_left=[[168,491],[225,510],[232,584],[183,693],[130,807],[76,914],[26,842],[47,772],[83,689],[122,600]]
    side_right=[[364,490],[411,574],[449,678],[482,773],[498,848],[480,907],[436,828],[388,725],[350,593]]
    # The original leg completion extends behind the skirt. Copying the source
    # over that whole mask paints cloth onto the hidden thigh armour. Trace the
    # observed upper leg boundaries separately and place the covering side cloth
    # in front; keep the original completed armour underneath.
    thigh_outlines={
        'red_leg_image_left':[[216,541],[224,559],[237,581],[248,602],[248,640],[233,675],
                              [222,699],[225,735],[205,757],[150,750],[137,717],[132,692],
                              [145,664],[162,627],[185,584],[205,553]],
        'red_leg_image_right':[[372,538],[389,579],[402,616],[415,658],[414,690],[403,717],
                               [383,747],[348,746],[330,717],[333,687],[346,665],[352,636],
                               [350,600],[363,576]],
        'blue_leg_image_left':[[733,542],[750,577],[764,599],[764,642],[745,684],[736,704],
                               [734,734],[711,756],[662,748],[650,719],[649,690],[662,657],
                               [685,610],[711,565]],
        'blue_leg_image_right':[[887,541],[903,579],[917,620],[929,660],[925,692],[913,721],
                                [905,743],[867,747],[849,717],[847,690],[862,662],[864,637],
                                [864,602],[878,577]],
    }
    for name,offset in [('red',0),('blue',514)]:
        scope=(xx<512) if offset==0 else (xx>=512)
        def moved(points: list[list[int]]) -> np.ndarray:
            return polygon(shape,[[x+offset,y] for x,y in points])
        h=_cut(head,head.mask & scope,f'{name}_head',role='original',parent=f'{name}_torso',
               meta={'label':f'{"红" if name=="red" else "蓝"}色头盔'})
        h.depth[:]=40;result.append(h)
        torso_mask=top.mask & scope & (yy<=500)
        torso=_cut(top,torso_mask,f'{name}_torso',role='original',
                   meta={'label':f'{"红" if name=="red" else "蓝"}色躯干铠甲'})
        torso.depth[:]=80;result.append(torso)
        skirt=top.mask & scope & (yy>=435) & target
        front=(moved(centre)|moved(hip_left)|moved(hip_right)) & skirt
        front_rgba=top.rgba.copy();front_rgba[~front]=0
        result.append(Layer(f'{name}_skirt_front',front_rgba,np.full(shape,100,np.uint8),'topwear',
                            role='cloth_outer',pivot=[300+offset,480],parent=f'{name}_torso',
                            meta={'label':f'{"红" if name=="red" else "蓝"}色前裙甲'}))
        legs=[p.mask for p in limb if p.name.startswith(name+'_leg')]
        leg_mask=np.logical_or.reduce(legs)
        visible_legs=np.logical_or.reduce([polygon(shape,points) for key,points in thigh_outlines.items()
                                           if key.startswith(name+'_leg')])
        visible_legs|=leg_mask & (yy>=680)
        covered_thigh=leg_mask & ~visible_legs & (yy<690)
        for side,points in [('left',side_left),('right',side_right)]:
            half=(xx<300+offset) if side=='left' else (xx>=300+offset)
            support=((moved(points)&skirt)|(covered_thigh&half)) & ~visible_legs & target
            # The author's garment prediction already contains clean cloth
            # behind the thighs. Reuse it rather than seed from leg pixels.
            known=support & top.mask & ~front
            rgba,completed=_complete_rgb(top.rgba,known,support,use_lama)
            result.append(Layer(f'{name}_skirt_side_{side}',rgba,np.full(shape,130,np.uint8),'topwear',
                                role='cloth_side',pivot=[190+offset if side=='left' else 370+offset,480],
                                parent=f'{name}_torso',meta={'label':f'{"红" if name=="red" else "蓝"}色{"左" if side=="left" else "右"}侧裙片',
                                                          'inferred_hidden_pixels':int(completed.sum())}))
        # The dark, undecorated lower arc is visible below the ornate front
        # apron. Use that lining material, rather than duplicating its embroidery.
        support=skirt & (yy>=490)
        known=support & (yy>=815) & (yy<=858) & (xx>=180+offset) & (xx<=410+offset) & ~leg_mask & ~front
        if known.sum()<30:
            raise ValueError(f'No reliable visible lining sample: {name}')
        rgba,completed=_complete_rgb(src,known,support,use_lama)
        result.append(Layer(f'{name}_skirt_inner',rgba,np.full(shape,230,np.uint8),'topwear',
                            role='cloth_inner',parent=f'{name}_torso',pivot=[300+offset,490],
                            meta={'label':f'{"红" if name=="red" else "蓝"}色内层裙片',
                                  'completion':'visible_dark_lining_and_author_LaMa',
                                  'inferred_hidden_pixels':int(completed.sum())}))
    return preserve_visible_source(src,result)
