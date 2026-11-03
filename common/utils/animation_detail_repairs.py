"""Source-guided fur edges and accessory ownership for reviewed deliveries."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from utils.animation_parts import Layer, _complete_rgb
from utils.animation_standard import curved_region, ellipse, reset_records
from utils.animation_integrity import polygon
from utils.animation_character_repairs import ink_path, _dilate, _largest, _over, _reviewed_layer


def _cat_foreground(src: np.ndarray) -> np.ndarray:
    """Refine a narrow, reviewed silhouette band; reject old background spikes."""
    shape=src.shape[:2]
    contour=[('M',315,226),('C',344,239,382,242,416,244),
        ('C',459,215,491,221,515,228),('C',540,214,566,226,593,239),
        ('C',628,207,669,184,695,163),('C',710,167,708,227,704,267),
        ('C',745,302,762,342,760,385),('C',755,420,760,454,766,478),
        ('C',773,514,774,550,780,582),('C',786,599,790,612,794,626),('C',810,687,800,762,788,807),
        ('C',771,859,749,900,716,924),('C',726,947,694,958,663,954),
        ('C',607,954,549,947,505,947),('C',476,961,434,956,402,953),
        ('C',367,954,327,946,299,925),('C',264,906,250,865,249,821),
        ('C',233,771,225,719,235,667),('C',241,616,278,583,327,566),
        ('C',343,553,339,528,337,510),('C',333,483,326,460,333,426),
        ('C',337,393,339,366,342,342),('C',330,311,324,270,315,226),('Z',)]
    initial=ink_path(shape,contour,[255,255,255])[...,3]>127
    trimap=np.zeros(shape,np.uint8)
    trimap[_dilate(initial,9)]=cv2.GC_PR_BGD
    trimap[initial]=cv2.GC_PR_FGD
    trimap[cv2.erode(initial.astype(np.uint8),np.ones((17,17),np.uint8))>0]=cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(src[...,:3].copy(),trimap,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
    mask=_largest((trimap==cv2.GC_FGD)|(trimap==cv2.GC_PR_FGD))
    mask&=initial
    mask=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    yy,xx=np.indices(shape)
    mask[(yy>=954)|((yy>=946)&(xx>499)&(xx<605))]=0
    return cv2.GaussianBlur(mask.astype(np.float32)*255,(0,0),.8).round().astype(np.uint8)


def _textured_part(src: np.ndarray, known: np.ndarray, support: np.ndarray,
                    name: str, role: str, depth: int, parent: str | None,
                    pivot: list[float], label: str, use_lama: bool,
                    foreground: np.ndarray | None=None) -> Layer:
    known=known&support
    rgba,missing=_complete_rgb(src,known,support,False)
    if use_lama and missing.sum()>20:
        from annotators.lama_inpainter import apply_inpaint
        y,x=np.where(support)
        x1,x2=max(0,int(x.min())-32),min(src.shape[1],int(x.max())+33)
        y1,y2=max(0,int(y.min())-32),min(src.shape[0],int(y.max())+33)
        sl=np.s_[y1:y2,x1:x2]
        material=apply_inpaint(src[sl][...,:3].copy(),missing[sl].astype(np.uint8)*255,
                               precision=__import__('torch').float32)
        rgba[sl][missing[sl],:3]=material[missing[sl]]
    alpha=cv2.GaussianBlur(support.astype(np.float32)*255,(0,0),.65)
    if foreground is not None:
        observed_boundary=_dilate(~(foreground>127),3)
        alpha[observed_boundary]=np.minimum(alpha[observed_boundary],foreground[observed_boundary])
    # Feathered alpha must carry edge material, not zero RGB; otherwise each
    # cut acquires a dark halo when the complete PNG is composited.
    edge=(alpha>0)&~support
    if edge.any():
        nearest=ndi.distance_transform_edt(~support,return_distances=False,return_indices=True)
        rgba[edge,:3]=rgba[nearest[0,edge],nearest[1,edge],:3]
    rgba[...,3]=np.round(alpha).astype(np.uint8);rgba[rgba[...,3]==0]=0
    part=_reviewed_layer(name,rgba,role,depth,parent,pivot,label)
    part.completed=missing
    return part


def rebuild_seated_cat(src: np.ndarray, layers: list[Layer], info: dict,
                       use_lama: bool=True) -> tuple[list[Layer],dict]:
    shape=src.shape[:2];yy,xx=np.indices(shape)
    foreground=_cat_foreground(src);fg=foreground>127
    outlines={
        'cat_body':[[373,557],[400,519],[511,494],[621,496],[708,508],[756,540],[775,600],[740,743],
            [709,866],[675,934],[578,950],[464,944],[379,908],[333,820],[323,691],[341,601],[373,557]],
        'cat_head':[[353,352],[376,293],[418,253],[459,230],[512,230],[552,228],
            [600,247],[659,273],[710,304],[744,346],[758,399],[751,465],[712,516],
            [640,546],[547,561],[442,548],[370,518],[337,467],[334,406],[353,352]],
        'cat_ear_left':[[315,227],[347,239],[387,243],[418,260],[424,290],[404,321],
            [375,353],[348,366],[338,336],[327,296],[315,227]],
        'cat_ear_right':[[590,256],[623,210],[667,180],[695,163],[707,188],
            [707,235],[702,272],[679,305],[635,307],[602,287],[590,256]],
        'cat_front_left':[[366,655],[394,635],[434,635],[463,654],[477,713],[481,790],
            [483,850],[495,892],[500,927],[484,950],[452,958],[413,952],
            [394,935],[377,892],[368,864],[350,810],[349,734],[352,690],[366,655]],
        'cat_front_right':[[675,680],[700,685],[716,716],[716,757],[697,803],
            [686,842],[680,876],[691,906],[688,930],[669,951],[637,951],[607,941],
            [602,918],[611,876],[626,830],[636,781],[646,733],[653,696],[675,680]],
        'cat_hind_left':[[300,574],[269,584],[244,618],[234,675],[231,733],
            [245,794],[251,845],[265,892],[288,924],[324,944],[370,946],
            [346,904],[345,864],[354,815],[358,749],[365,698],[349,641],[320,587],[300,574]],
        'cat_hind_right':[[740,609],[766,607],[790,649],[792,693],[797,755],
            [785,832],[760,878],[731,924],[707,936],[699,923],[708,886],
            [715,847],[722,803],[721,751],[718,698],[718,649],[740,609]],
    }
    supports={n:curved_region(shape,points) for n,points in outlines.items()}
    parts=[];source_masks={}
    features={
        'cat_eye_left':(ellipse(shape,[459,366,46,42]),[459,366],'左眼'),
        'cat_eye_right':(ellipse(shape,[629,345,39,38]),[629,345],'右眼'),
        'cat_nose':(curved_region(shape,[[532,381],[549,377],[567,383],[563,398],[551,410],[541,401],[532,381]]),[550,390],'鼻子')}
    head_holes=np.zeros(shape,bool)
    for n,(mask,pivot,label) in features.items():
        mask&=fg;rgba=src.copy();rgba[~mask]=0
        core=mask
        if n.startswith('cat_eye'):
            core=ellipse(shape,[*pivot,39 if n.endswith('left') else 32,
                                34 if n.endswith('left') else 32])&fg
            distance=ndi.distance_transform_edt(~core)
            rgba[...,3]=np.round(np.clip(1-distance/8,0,1)*255).astype(np.uint8)
            rgba[~mask]=0
        parts.append(_reviewed_layer(n,rgba,'face_feature',10,'cat_head',pivot,label))
        head_holes|=_dilate(mask,9)
        source_masks[n]=core
    definitions=[
        ('cat_ear_left','source_region',110,'cat_head',[367,308],'画面左耳'),
        ('cat_ear_right','source_region',110,'cat_head',[651,271],'画面右耳'),
        ('cat_front_left','leg_whole',50,'cat_body',[408,683],'画面左前腿'),
        ('cat_front_right','leg_whole',50,'cat_body',[678,715],'画面右前腿'),
        ('cat_hind_left','leg_whole',160,'cat_body',[306,663],'画面左后腿与臀部'),
        ('cat_hind_right','leg_whole',160,'cat_body',[752,688],'画面右后腿可见部分'),
        ('cat_head','animal_head',30,'cat_body',[545,511],'头部'),
        ('cat_body','animal_body',130,None,[545,701],'身体')]
    for n,role,depth,parent,pivot,label in definitions:
        support=supports[n]
        # Only the actual outer silhouette limits visible fur. Root caps stay
        # inside that silhouette and overlap their parent rather than floating.
        support&=fg
        known=support.copy()
        if n.startswith('cat_ear'):
            known&=~_dilate(supports['cat_head'],5)
        elif n.startswith('cat_front'):
            # The short visible forelegs already contain their own fur at the
            # root; do not replace that real texture with a solid colour cap.
            pass
        elif n.startswith('cat_hind'):
            known&=~_dilate(supports['cat_front_left']|supports['cat_front_right'],8)
        elif n=='cat_head':
            known&=~head_holes&~_dilate(supports['cat_ear_left']|supports['cat_ear_right'],5)
        else:
            occluders=supports['cat_head']|supports['cat_front_left']|supports['cat_front_right']
            known&=~_dilate(occluders,12)
        p=_textured_part(src,known,support,n,role,depth,parent,pivot,label,use_lama,foreground)
        if n=='cat_head':
            # LaMa can leave an eyelid-like crease after eye removal. Continue
            # adjacent real face fur into the underlay instead of retaining it.
            for target,donor,radii in [([459,366],[390,473],[58,52]),
                                        ([629,345],[692,450],[50,49])]:
                rx,ry=radii;dx,dy=donor;tx,ty=target
                patch=src[dy-ry:dy+ry+1,dx-rx:dx+rx+1,:3].copy()
                mask=ellipse(patch.shape[:2],[rx,ry,rx-2,ry-2]).astype(np.uint8)*255
                cloned=cv2.seamlessClone(patch,p.rgba[...,:3].copy(),mask,(tx,ty),cv2.NORMAL_CLONE)
                selected=ellipse(shape,[tx,ty,rx-2,ry-2])
                p.rgba[selected,:3]=cloned[selected]
        source_masks[n]=support;parts.append(p)
    # A clean rest pose uses real source material wherever its anatomical part
    # is observed. Completed RGB is never overwritten with an occluder's pixels.
    by={p.name:p for p in parts}
    owner=np.full(shape,-1,np.int16)
    order=['cat_hind_left','cat_hind_right','cat_body','cat_front_left','cat_front_right',
           'cat_ear_left','cat_ear_right','cat_head','cat_eye_left','cat_eye_right','cat_nose']
    for i,n in enumerate(order):owner[source_masks[n]]=i
    unowned=fg&(owner<0)
    if unowned.any():
        fields=np.stack([ndi.distance_transform_edt(~source_masks[n])[unowned] for n in order])
        owner[unowned]=fields.argmin(axis=0)
    for i,n in enumerate(order):
        observed=fg&(owner==i)
        # Preserve the feathered completion band. Restoring source RGB across
        # it would paste the eye rim / sleeve outline back into a clean part.
        extra=observed&~by[n].mask
        by[n].rgba[extra]=src[extra]
        # Ownership masks meet without overlap. Internal antialiasing would
        # multiply two soft boundaries and leave a transparent seam.
        if n!='cat_head' and n not in features:
            by[n].rgba[observed]=src[observed]
        if n not in features:by[n].rgba[observed,3]=255
        by[n].visibility_mask=np.where(observed,foreground,0).astype(np.uint8)
        if n in features:
            # The fur feather around an eye blends over the head, while its
            # opaque iris owns the core. Disjoint soft masks would leave a ring.
            by[n].visibility_mask=foreground.copy()
    return parts,reset_records(info,[{'name':'cat','kind':'seated_quadruped',
        'parts':[p.name for p in parts],'inferred_hidden_parts':[]}])


def repair_satchel(src: np.ndarray, layers: list[Layer], info: dict,
                    use_lama: bool=True) -> tuple[list[Layer],dict]:
    by={p.name:p for p in layers};objects=by['objects'];clothes=by['clothing']
    shape=src.shape[:2];yy,xx=np.indices(shape)
    bag_mask=objects.mask&(xx<520);round_mask=objects.mask&~bag_mask
    strap_region=polygon(shape,[[477,332],[497,334],[490,365],[481,407],
        [468,449],[453,469],[447,460],[458,409],[468,365],[477,332]])
    visible=strap_region&clothes.mask
    # Source brown belt and gold buckle belong to the bag, including their rim.
    material=(clothes.rgba[...,0].astype(int)>clothes.rgba[...,2].astype(int)+12)
    material&=clothes.rgba[...,1]<170
    visible&=_dilate(material,1)
    strap=ink_path(shape,[('M',477,333),('C',485,327,496,330,495,342),
        ('C',483,416,456,473,415,521),('C',409,531,395,531,389,518),
        ('C',424,477,457,417,469,351),('C',472,341,472,336,477,333),('Z',)],
        [91,58,43],[40,34,30],2)
    strap[visible]=clothes.rgba[visible]
    bag_rgba=objects.rgba.copy();bag_rgba[~bag_mask]=0
    bag_rgba=_over(strap,bag_rgba)
    bag=_reviewed_layer('satchel',bag_rgba,'accessory',int(objects.z),
        'clothing',[486,342],'挎包与包带')
    # The sleeve passes over the strap in the original pose. The complete PNG
    # keeps the connection for rigging, with reversible rest-pose occlusion.
    bag_visibility=bag_mask|visible
    if objects.visibility_mask is not None:
        bag_visibility=(bag_mask&(objects.visibility_mask>0))|visible
    bag.visibility_mask=bag_visibility.astype(np.uint8)*255
    round_rgba=objects.rgba.copy();round_rgba[~round_mask]=0
    accessory=_reviewed_layer('round_accessory',round_rgba,'accessory',int(objects.z),
        'clothing',[677,443],'圆形配件')
    accessory.visibility_mask=objects.visibility_mask
    known=clothes.mask&~_dilate(visible,4)
    support=clothes.mask.copy()
    clothes.rgba,clothes.completed=_complete_rgb(clothes.rgba,known,support,use_lama)
    if clothes.visibility_mask is not None:clothes.visibility_mask[visible]=0
    for name in ('arm_image_left_upper','arm_image_left_forearm'):
        arm=by[name]
        if arm.visibility_mask is None:arm.visibility_mask=np.full(shape,255,np.uint8)
        # Those inferred sleeve pixels were originally covered by the belt in
        # the garment layer. Moving the belt must preserve that occlusion.
        arm.visibility_mask[visible]=0
    clothes.meta['accessory_cleanup']='brown satchel strap moved into satchel layer; garment behind it completed'
    result=[p for p in layers if p.name!='objects']+[bag,accessory]
    info['accessory_groups']=[{'name':'satchel','parts':['bag body','observed brown strap','bounded hidden connection'],
                              'provenance':'reviewed source strap belongs to the left satchel'}]
    return result,info
