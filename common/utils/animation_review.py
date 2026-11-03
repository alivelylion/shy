"""Asset-specific semantic repairs for the reviewed compact animation delivery."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from utils.animation_parts import Layer, _complete_rgb, _cut
from utils.animation_integrity import polygon
from utils.animation_special import preserve_visible_source


def largest(mask: np.ndarray) -> np.ndarray:
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return labels==(1+stats[1:,cv2.CC_STAT_AREA].argmax()) if n>1 else mask.copy()


def ground_shadow(shape: tuple[int,int], centre: tuple[float,float], radii: tuple[float,float],
                  name: str, opacity: int = 140) -> Layer:
    """A complete soft ground oval, without a character-shaped cutout."""
    yy,xx=np.indices(shape)
    r=((xx-centre[0])/radii[0])**2+((yy-centre[1])/radii[1])**2
    alpha=np.round(opacity*np.exp(-2.4*r)).astype(np.uint8)
    alpha[r>1.5]=0
    rgba=np.zeros((*shape,4),np.uint8);rgba[...,:3]=[35,30,32];rgba[...,3]=alpha
    return Layer(name,rgba,np.full(shape,250,np.uint8),'reviewed:ground_shadow',role='source_effect',
                 meta={'label':'椭圆接地阴影','provenance':'soft_complete_ground_oval'})


def _feature_rgba(src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgba=src.copy();rgba[~mask]=0
    return rgba


def _blend_underpaint(rgba: np.ndarray, covered: np.ndarray, material: np.ndarray,
                       protect: np.ndarray | None = None) -> None:
    distance=ndi.distance_transform_edt(~covered)
    weight=np.clip(1-distance/18,0,1)
    weight=weight*weight*(3-2*weight)
    if protect is not None:
        weight[protect & ~covered]=0
    weight[rgba[...,3]<=10]=0
    rgba[...,:3]=np.round(rgba[...,:3]*(1-weight[...,None])+material*weight[...,None]).astype(np.uint8)


def _repair_pikachu(src: np.ndarray, layers: list[Layer], ash: bool) -> list[Layer]:
    prefix='ash_pikachu' if ash else 'pikachu'
    by={p.name:p for p in layers};shape=src.shape[:2];yy,xx=np.indices(shape)
    head=by[prefix+'_head']
    boxes=([(545,267,628,343),(724,278,805,353)] if ash else
           [(693,495,792,610),(883,510,962,608)])
    eyes=np.zeros(shape,bool)
    rgb=src[...,:3].astype(np.int16)
    for x1,y1,x2,y2 in boxes:
        candidate=np.zeros(shape,bool)
        candidate[y1:y2,x1:x2]=(rgb[y1:y2,x1:x2].max(axis=-1)<210)&(rgb[y1:y2,x1:x2,2]<130)
        selected=largest(candidate & (src[...,3]>10))
        ys,xs=np.where(selected)
        if len(xs):
            hull=cv2.convexHull(np.column_stack([xs,ys]).astype(np.int32))
            filled=np.zeros(shape,np.uint8);cv2.fillConvexPoly(filled,hull,1)
            eyes|=filled.astype(bool)
    eyes=cv2.dilate(eyes.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)&(src[...,3]>10)
    by['eyes'].rgba=_feature_rgba(src,eyes)
    by['eyes'].parent=head.name;by['eyes'].meta['label']='眼睛'
    feature_boxes={'mouth':(633,371,744,402),'nose':(689,335,722,364)} if ash else {'mouth':(772,636,892,740)}
    for name,(x1,y1,x2,y2) in feature_boxes.items():
        candidate=np.zeros(shape,bool)
        candidate[y1:y2,x1:x2]=(rgb[y1:y2,x1:x2,1]<155)&(rgb[y1:y2,x1:x2,2]<150)
        candidate&=src[...,3]>10
        candidate=ndi.binary_fill_holes(candidate if ash and name=='mouth' else largest(candidate))
        candidate=cv2.dilate(candidate.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        by[name].rgba=_feature_rgba(src,candidate)
    if not ash:
        by['nose'].rgba=_feature_rgba(src,by['nose'].mask)
    else:
        hat=by[prefix+'_hat']
        hat_region=polygon(shape,[[442,0],[778,0],[837,104],[864,225],[905,274],
                                  [905,300],[807,279],[587,251],[393,239],[416,194]])
        hat_material=(rgb[...,2]>130)|((rgb[...,0]>100)&(rgb[...,1]<rgb[...,0]*.75))
        hm=ndi.binary_fill_holes(largest(hat_region&hat_material&(src[...,3]>10)))
        hm=cv2.dilate(hm.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&hat_region&(src[...,3]>10)
        hat.rgba=_feature_rgba(src,hm)
    repair=cv2.dilate(eyes.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
    for part in ('nose','mouth'):
        if part in by:
            repair|=cv2.dilate(by[part].mask.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
    # Include the larger rings left by the earlier prediction around the eyes.
    for x1,y1,x2,y2 in boxes:
        cv2.ellipse(repair.view(np.uint8),((x1+x2)//2,(y1+y2)//2),
                    ((x2-x1)//2+12,(y2-y1)//2+12),0,0,360,1,-1)
    if ash:
        repair|=head.mask & (yy<280) & ((head.rgba[...,2]>130)|(head.rgba[...,0]<120))
        head_outline=[[441,220],[591,208],[783,227],[822,277],[828,409],[792,459],
                      [746,488],[519,495],[467,460],[439,405]]
        ears={prefix+'_ear_left':[[180,239],[252,197],[355,177],[460,183],[534,215],
                                 [509,248],[398,272],[262,278]],
              prefix+'_ear_right':[[810,263],[849,280],[870,339],[875,404],[855,480],
                                  [826,453],[813,384]]}
    else:
        head_outline=[[552,453],[600,430],[634,390],[693,360],[775,347],[856,371],
                      [937,425],[971,505],[980,593],[965,681],[923,729],[825,769],
                      [700,767],[585,713],[549,624],[532,529]]
        ears={prefix+'_ear_near':[[348,263],[434,272],[530,315],[610,361],[669,411],
                                 [618,434],[518,408],[424,362],[375,311]],
              prefix+'_ear_far':[[798,81],[850,93],[901,157],[941,258],[956,368],
                                [958,445],[901,412],[850,324],[821,224]]}
    original_head=head.mask
    support=ndi.binary_fill_holes(largest(original_head & polygon(shape,head_outline)))
    repair|=support & ~original_head
    repair|=support & (yy<(285 if ash else 520)) & (head.rgba[...,0]<180)
    repair|=support & (((xx<480)&(yy<305))|((xx>805)&(yy<430)) if ash else
                       ((xx<650)&(yy<530)))
    repair&=support
    yellow=(head.rgba[...,0]>190)&(head.rgba[...,1]>145)&(head.rgba[...,2]<150)
    trusted=support & ~repair & yellow
    if not trusted.any():
        raise ValueError('No clean yellow face material remains')
    nearest=ndi.distance_transform_edt(~trusted,return_distances=False,return_indices=True)
    material=head.rgba[...,:3][tuple(nearest)].copy()
    material=cv2.GaussianBlur(material,(0,0),8)
    head.rgba[repair,:3]=material[repair]
    head.rgba[support & ~original_head,3]=255
    head.rgba[~support]=0
    head.meta['reviewed_face_base']='eye, nose and mouth rings removed beneath independent features'
    for name,points in ears.items():
        p=by[name]
        mask=largest(polygon(shape,points)&(src[...,3]>10))
        cheek=(rgb[...,0]>100)&(rgb[...,1]<rgb[...,0]*.6)
        p.rgba,_=_complete_rgb(src,mask & ~cheek,mask,False)
        p.parent=head.name;p.meta['reviewed_ear_boundary']=True
    for p in layers:
        if p.name.startswith(prefix) and p.role in {'arm_whole','leg_whole','animal_body'}:
            lower=p.mask & (yy>(820 if ash else 750))
            yellow=lower & (p.rgba[...,0]>100) & (p.rgba[...,1]>65)
            clean=ndi.binary_fill_holes(yellow)
            p.rgba[lower & ~clean]=0
            p.rgba[~largest(p.mask)]=0
    shadow=ground_shadow(shape,(625,934) if ash else (582,854),
                         (310,74) if ash else (427,60),prefix+'_shadow',140)
    parts=[p for p in layers if p.name!=shadow.name]
    for p in parts:
        if p.name in {'eyes','nose','mouth'}:p.depth[:]=15
        elif p.name==prefix+'_hat':p.depth[:]=20
        elif '_ear_' in p.name:p.depth[:]=110
        elif p.name==prefix+'_head':p.depth[:]=90
        elif p.role=='animal_body':p.depth[:]=160 if ash else 80
        elif p.role in {'arm_whole','leg_whole'}:p.depth[:]=65
        elif p.role=='animal_tail':p.depth[:]=200
    target=src[...,3]>10
    lower=yy>(820 if ash else 750)
    yellow=target & lower & (src[...,0]>100)&(src[...,1]>65)
    target[lower]=ndi.binary_fill_holes(yellow)[lower]
    _,labels,stats,_=cv2.connectedComponentsWithStats(target.astype(np.uint8),8)
    keep=np.zeros(len(stats),bool);keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=32
    target=keep[labels]
    parts=preserve_visible_source(src,parts,target)
    hidden_features=by['eyes'].mask|by['nose'].mask|by['mouth'].mask
    if ash:
        hidden_features|=by[prefix+'_hat'].mask
    else:
        dark_back=(xx<650)&(yy>=425)&(yy<530)&(rgb[...,0]<170)&(rgb[...,1]<140)&(rgb[...,2]<100)&target
        by[prefix+'_body'].rgba[dark_back]=src[dark_back]
        hidden_features|=dark_back
    cheek=(rgb[...,0]>100)&(rgb[...,1]<rgb[...,0]*.6)&(yy>(325 if ash else 550))
    _blend_underpaint(head.rgba,hidden_features & head.mask,material,cheek)
    return [*parts,shadow]


def _animal_sheet(src: np.ndarray) -> list[Layer]:
    definitions=[('chicken_left','左上鸡',(100,100,360,355)),
                 ('chicken_middle','中上鸡',(395,100,640,355)),
                 ('cow_upper','右上牛',(650,95,945,355)),
                 ('cow_middle','左中牛',(85,385,365,640)),
                 ('sheep_middle','右中羊',(650,375,940,650)),
                 ('sheep_lower','左下羊',(85,675,365,935)),
                 ('pig','小猪',(380,675,650,935)),
                 ('duck','小鸭',(675,665,940,940))]
    result=[]
    for name,label,(x1,y1,x2,y2) in definitions:
        mask=np.zeros(src.shape[:2],bool);mask[y1:min(y2,930),x1:x2]=src[y1:min(y2,930),x1:x2,3]>10
        rgba=_feature_rgba(src,mask)
        result.append(Layer(name,rgba,np.full(mask.shape,100,np.uint8),'reviewed:animal_sheet',
                            role='animal_whole',meta={'label':label,'provenance':'complete_source_sprite'}))
    union=np.logical_or.reduce([p.mask for p in result])
    missing=(src[...,3]>10)&~union
    if missing.any():
        if np.where(missing)[0].min()<930:
            raise ValueError('The reviewed animal cells do not cover the complete sheet')
        result.append(Layer('source_text',_feature_rgba(src,missing),np.full(missing.shape,100,np.uint8),
                            'reviewed:animal_sheet',role='graphic_text',meta={'label':'原图文字与水印'}))
    return result


def _repair_doraemon(src: np.ndarray, layers: list[Layer]) -> list[Layer]:
    by={p.name:p for p in layers};shape=src.shape[:2];yy,xx=np.indices(shape)
    # Keep the reviewed foreground silhouettes; replace blurred generated
    # visible texture with the actual line art rather than sharpening artifacts.
    for p in layers:
        if p.name.startswith('doraemon'):
            mask=largest(p.mask)
            edge=cv2.GaussianBlur(mask.astype(np.float32),(3,3),.6)
            alpha=np.round(np.clip((edge-.15)/.7,0,1)*255).astype(np.uint8)
            p.rgba=src.copy();p.rgba[...,3]=alpha;p.rgba[alpha==0]=0
    head=by['doraemon_head'];old_eyes=by['eyes'].mask.copy()
    neck_edge=np.interp(xx,[290,390,490,590,670],[499,520,536,545,549])
    head.rgba[(xx>=290)&(xx<=670)&(yy>neck_edge)]=0
    eye_mask=np.zeros(shape,bool)
    white=(src[...,0]>210)&(src[...,1]>210)&(src[...,2]>200)
    for (x1,y1,x2,y2),seed in [((340,40,487,218),(420,90)),((461,74,611,242),(530,130))]:
        candidate=np.zeros(shape,np.uint8);candidate[y1:y2,x1:x2]=white[y1:y2,x1:x2]
        _,labels,_,_=cv2.connectedComponentsWithStats(candidate,8)
        component=labels[seed[1],seed[0]]
        if component==0:
            raise ValueError('Reviewed eye seed is outside the white eyeball')
        patch=ndi.binary_fill_holes(labels==component)
        patch=cv2.dilate(patch.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
        eye_mask|=patch
    eye_mask&=head.mask
    by['eyes'].rgba=_feature_rgba(src,eye_mask);by['eyes'].parent=head.name
    eye_red=eye_mask&(src[...,0]>110)&(src[...,1]<100)&(src[...,2]<160)
    by['eyes'].rgba[eye_red,:3]=[248,249,246]
    by['eyes'].meta['label']='眼睛';by['eyes'].depth[:]=35
    nose=by['nose'];nose.rgba=_feature_rgba(src,nose.mask)
    nose.depth[:]=25;nose.parent=head.name
    face_outline=[[375,124],[327,139],[276,168],[237,205],[217,258],[214,302],
                  [225,352],[252,415],[300,475],[363,509],[451,531],[556,543],
                  [658,548],[704,493],[726,442],[729,397],[715,343],[684,284],
                  [651,237],[601,202],[550,180],[480,151],[411,129]]
    face=polygon(shape,face_outline)&head.mask
    inside=cv2.erode(face.astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)
    corridor=np.zeros(shape,np.uint8)
    paths=[[[276,132],[370,218]],[[232,209],[355,250]],[[212,289],[351,286]],
           [[524,263],[695,243]],[[522,294],[699,336]],[[521,318],[665,427]],
           [[306,261],[296,245],[274,236],[253,242],[236,264],[231,287],[239,323],
            [282,361],[349,391],[428,407],[509,404],[579,385],[648,352]],
           [[432,249],[412,307],[395,360],[383,396]]]
    cv2.polylines(corridor,[np.asarray(p,np.int32) for p in paths],False,1,13)
    ink=(src[...,:3].max(axis=-1)<180)&(inside|(corridor>0))&~eye_mask&~nose.mask
    ink&=(yy>125)&(yy<450)&(xx>215)&(xx<710)
    ink=cv2.dilate(ink.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&head.mask&~eye_mask&~nose.mask
    face_rgba=_feature_rgba(src,ink)
    lines=Layer('doraemon_face_lines',face_rgba,np.full(shape,40,np.uint8),'source:doraemon',
                role='face_feature',parent=head.name,meta={'label':'嘴与胡须线条'})
    feature=old_eyes|nose.mask|eye_mask|ink
    clear=cv2.dilate(feature.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)&head.mask
    head.rgba[clear & face,:3]=[248,249,246]
    head.rgba[clear & ~face,:3]=[0,163,215]
    red_in_head=head.mask&(src[...,0]>110)&(src[...,1]<100)&(src[...,2]<160)
    head.rgba[red_in_head & face,:3]=[248,249,246]
    head.rgba[red_in_head & ~face,:3]=[0,163,215]
    head.depth[:]=80
    body=by['doraemon_body'];body.rgba[yy>846]=0;body.depth[:]=170
    body.rgba[body.mask & (yy<580),:3]=[0,163,215]
    arm=by['doraemon_folded_arm'];arm.depth[:]=100
    behind=body.mask & cv2.dilate(arm.mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    body.rgba[behind,:3]=[0,163,215]
    # Continue the torso under the leg roots instead of keeping a duplicated foot.
    for name in ['doraemon_leg_left','doraemon_leg_right']:
        by[name].depth[:]=130
    collar=by['doraemon_collar'];collar.depth[:]=70
    red=(src[...,0]>110)&(src[...,1]<100)&(src[...,2]<160)
    collar_region=polygon(shape,[[282,492],[675,532],[675,583],[281,547]])
    cm=largest(red&collar_region)
    cm=cv2.dilate(cm.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&collar_region
    collar.rgba=_feature_rgba(src,cm)
    tail=by['doraemon_tail'];tm=largest(red&(xx>665)&(yy>730)&(yy<830))
    tm=cv2.dilate(tm.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    connector=np.zeros(shape,np.uint8);cv2.line(connector,(660,776),(681,781),1,3)
    tail.rgba=_feature_rgba(src,tm|(connector>0));tail.depth[:]=180
    parts=[p for p in layers if p.name!='doraemon_face_lines']+[lines]
    # The actual black perimeter separates the blue character from its blue
    # background more precisely than a hand-drawn polygon does.
    barrier=(src[...,:3].max(axis=-1)<160)&(xx>195)&(xx<825)&(yy>30)&(yy<1006)
    barrier=cv2.morphologyEx(barrier.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(barrier,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    target=np.zeros(shape,np.uint8)
    cv2.drawContours(target,[max(contours,key=cv2.contourArea)],-1,1,-1)
    if target.sum()<250000:
        raise ValueError('The source line-art perimeter is not closed')
    parts=preserve_visible_source(src,parts,target.astype(bool))
    # Source restoration must not put an antialiased copy of the facial ink
    # back into the base underneath its independent feature layer.
    covered=(by['eyes'].mask|nose.mask|lines.mask)&head.mask
    material=np.zeros_like(head.rgba[...,:3]);material[:]=[0,163,215];material[face]=[248,249,246]
    boundary=ndi.distance_transform_edt(head.mask)<=4
    _blend_underpaint(head.rgba,covered,material,boundary)
    head.rgba[inside & head.mask,:3]=[248,249,246]
    return parts


def _repair_cat(layers: list[Layer]) -> list[Layer]:
    by={p.name:p for p in layers};leg=by['cat_front_right'];body=by['cat_body']
    outline=[[663,690],[700,694],[716,744],[708,799],[695,845],[683,892],
             [687,927],[676,948],[619,955],[601,944],[609,911],[625,886],
             [638,843],[644,798],[647,748]]
    keep=largest(leg.mask & polygon(leg.mask.shape,outline))
    returned=leg.mask & ~keep
    body.rgba[returned]=leg.rgba[returned]
    leg.rgba[~keep]=0
    leg.meta.update({'reviewed_limb_boundary':True,'label':'右前肢整体'})
    return layers


def _separate_sheet_tail(src: np.ndarray, layers: list[Layer]) -> list[Layer]:
    by={p.name:p for p in layers};hair,tail=by['hair'],by['tail']
    _,labels,stats,_=cv2.connectedComponentsWithStats(hair.mask.astype(np.uint8),8)
    for i in range(1,len(stats)):
        component=labels==i
        if (component&tail.mask).sum()>max(4,component.sum()*.15):
            extra=component & ~tail.mask
            tail.rgba[extra]=hair.rgba[extra]
            hair.rgba[component]=0
    tail.rgba=_feature_rgba(src,tail.mask&(src[...,3]>10))
    tail.meta['label']='尾巴';tail.meta['removed_from_hair']=True
    return layers


def _separate_fur(layers: list[Layer], use_lama: bool) -> list[Layer]:
    definitions={
        'arm_component1':('right',[[540,180],[567,177],[608,215],[638,242],[655,281],
                                  [611,290],[577,259],[544,226]],
                          [[518,348],[539,347],[563,376],[558,412],[544,429],[523,418],[512,389]]),
        'arm_component2':('left',[[412,194],[453,193],[489,225],[468,255],[438,265],[411,238]],
                          [[267,390],[298,376],[322,410],[323,449],[305,482],[274,475],[257,449]])}
    yy,xx=np.indices(layers[0].mask.shape)
    hand=next((p for p in layers if p.name=='handwear'),None)
    if hand is not None and not any(p.name in definitions for p in layers):
        # Compact metadata need not retain the former, invalid coat skeletons.
        # The original two coat/arm components lie on opposite sides of x=500.
        left=_cut(hand,hand.mask&(xx<500),'arm_component2')
        right=_cut(hand,hand.mask&(xx>=500),'arm_component1')
        layers=[p for p in layers if p is not hand]+[left,right]
    result=[p for p in layers if p.name not in definitions]
    for p in layers:
        if p.name not in definitions:
            continue
        side,arm_points,glove_points=definitions[p.name]
        arm_mask=p.mask & polygon(p.mask.shape,arm_points)
        glove_mask=p.mask & polygon(p.mask.shape,glove_points)
        for mask,name,label in [(arm_mask,f'arm_visible_{side}','可见上臂与袖口'),
                                (glove_mask,f'glove_{side}','手套与手部')]:
            if mask.sum()>20:
                part=_cut(p,mask,name,role='arm_visible' if name.startswith('arm_') else 'hand',
                          parent='topwear',meta={'label':label,'provenance':'visible_regions_only'})
                part.depth[:]=75;result.append(part)
        support=p.mask & (~arm_mask if side=='left' else (yy>=263))
        trusted=support & ~cv2.dilate((arm_mask|glove_mask).astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
        rgba,completed=_complete_rgb(p.rgba,trusted,support,use_lama)
        coat=Layer(f'fur_panel_{side}',rgba,np.full(p.mask.shape,120,np.uint8),'handwear',
                   role='garment',parent='topwear',completed=completed,
                   meta={'label':f'画面{"右" if side=="right" else "左"}侧毛皮披肩',
                         'inferred_under_glove_pixels':int(completed.sum())})
        result.append(coat)
    for p in result:
        if p.name=='fur_panel_left':
            outline=[[440,199],[455,216],[450,250],[462,281],[488,311],[490,356],
                     [476,403],[458,449],[431,503],[404,556],[389,613],[392,690],
                     [392,776],[380,835],[359,877],[333,900],[296,909],[268,890],
                     [256,856],[255,795],[259,734],[267,655],[276,586],[286,520],
                     [312,500],[329,470],[332,442],[328,414],[319,386],[311,342],[315,297],[334,259],
                     [363,225],[404,203]]
            keep=largest(p.mask & polygon(p.mask.shape,outline))
            yy,xx=np.indices(p.mask.shape)
            rgb=p.rgba[...,:3].astype(np.int16)
            skin=(yy<290)&(xx>430)&(rgb[...,0]>150)&(rgb[...,0]>rgb[...,1]+18)&(rgb[...,0]>rgb[...,2]+30)
            keep&=~skin
            p.rgba[~keep]=0
            p.meta['glove_fringe_and_detached_skin_removed']=True
    return result


def _curved_joint_region(shape: tuple[int, int], start: list[float],
                         curves: list[list[list[float]]], tail: list[list[float]]) -> np.ndarray:
    """Rasterize a reviewed garment seam, rather than a straight bone bisector."""
    point = np.asarray(start, dtype=float)
    boundary = [point]
    t = np.linspace(0, 1, 81)[:, None]
    for controls in curves:
        c1, c2, end = np.asarray(controls, dtype=float)
        boundary.extend((1-t)**3*point + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*end)
        point = end
    boundary.extend(np.asarray(tail, dtype=float))
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.round(boundary).astype(np.int32)], 1)
    return result.astype(bool)


def _complete_round_joint(rgba: np.ndarray, known: np.ndarray, centre: tuple[float, float],
                          radius: float, material: np.ndarray,
                          bridge_from: tuple[float, float] | None = None) -> tuple[np.ndarray, int]:
    """Add a small, antialiased joint disk using only the selected material.

    The disk belongs behind the mating piece. Its texture is inferred only in
    the missing joint area; existing visible texture is copied without blur.
    """
    yy, xx = np.indices(known.shape)
    if bridge_from is None:
        distance = np.hypot(xx-centre[0], yy-centre[1])
    else:
        a = np.asarray(bridge_from, dtype=float)
        b = np.asarray(centre, dtype=float)
        xy = np.stack([xx, yy], axis=-1)
        along = np.clip(((xy-a)*(b-a)).sum(axis=-1)/max(float(((b-a)**2).sum()), 1), 0, 1)
        distance = np.linalg.norm(xy-(a+along[..., None]*(b-a)), axis=-1)
    coverage = np.clip(radius+.5-distance, 0, 1)
    added = (coverage > 0) & ~known
    result = _feature_rgba(rgba, known)
    if not added.any():
        return result, 0
    trusted = known & material & (rgba[..., 3] >= 220)
    if trusted.sum() < 12:
        raise ValueError('A rounded joint needs clean material from its own limb')
    nearest = ndi.distance_transform_edt(~trusted, return_distances=False, return_indices=True)
    texture = rgba[..., :3][tuple(nearest)]
    texture = cv2.GaussianBlur(texture, (0, 0), 2.5)
    # Continue the illustrated outline around the new hidden end, without
    # drawing a new ring over original skin or armour details.
    rim = np.clip((distance-(radius-3))/1.2, 0, 1)
    texture = np.round(texture*(1-rim[..., None]) + np.array([23, 23, 25])*rim[..., None]).astype(np.uint8)
    result[added, :3] = texture[added]
    result[added, 3] = np.round(255*coverage[added]).astype(np.uint8)
    return result, int(added.sum())


def _finish_engineer_arm_joints(layers: list[Layer]) -> list[Layer]:
    """Follow the sleeve opening and mechanical elbow on the engineer image."""
    by = {p.name: p for p in layers}
    names = [f'arm_image_{s}_{part}' for s in ('left', 'right') for part in ('upper', 'forearm')]
    if not all(n in by for n in names):
        raise ValueError('The reviewed engineer image requires both complete arm pairs')
    if all(by[n].meta.get('joint_finish') for n in names):
        return layers
    joined = np.zeros_like(by[names[0]].rgba)
    for name in names:
        p = by[name]
        choose = p.rgba[..., 3] > joined[..., 3]
        joined[choose] = p.rgba[choose]
    yy, xx = np.indices(joined.shape[:2])
    # The old nearest-bone assignment put a pointed sliver of the far
    # shoulder on top of the near glove. Assign the touching arms first.
    boundary_x = np.interp(yy, [0, 400, 425, 434, 440, 451, 469, 490, 510, 530, 560, 1024],
                           [500, 520, 520, 535, 551, 570, 586, 592, 585, 567, 550, 550])
    left_arm = xx <= boundary_x
    for side in ('left', 'right'):
        upper = by[f'arm_image_{side}_upper']
        fore = by[f'arm_image_{side}_forearm']
        # Rejoin the author's pixels first, so old diagonal cuts are discarded.
        rgba = _feature_rgba(joined, left_arm if side == 'left' else ~left_arm)
        full = rgba[..., 3] > 10
        yy, xx = np.indices(full.shape)
        if side == 'left':
            # The black/white cuff edge belongs to the sleeve. The skin cap
            # extends under it, instead of taking a triangular piece of cuff.
            distal = _curved_joint_region(full.shape, [445, 471], [
                [[418, 467], [391, 482], [384, 502]],
                [[376, 521], [381, 532], [399, 533]],
                [[411, 534], [424, 531], [435, 530]],
            ], [[610, 565], [610, 410], [470, 410], [470, 465]])
            end_x = np.interp(yy, [458, 468, 480, 495, 515, 530, 545, 560],
                             [466, 449, 452, 451, 445, 433, 409, 390])
            sleeve = (yy < 458) | (xx <= end_x)
            fore_known = full & (distal | ~sleeve)
            fore_known = largest(fore_known)
            upper_known = largest(full & sleeve & ~fore_known)
            rgb = rgba[..., :3].astype(np.int16)
            skin = (rgb[..., 0] > 170) & (rgb[..., 0] > rgb[..., 1]+12) & (rgb[..., 1] > 100)
            fore.rgba, count = _complete_round_joint(rgba, fore_known, (407, 508), 26, skin)
            white = (rgb.min(axis=-1) > 210) & (rgb.max(axis=-1)-rgb.min(axis=-1) < 25)
            upper.rgba, sleeve_count = _complete_round_joint(rgba, upper_known, (411, 505), 35, white)
            # Complete the solid sleeve end only inside the observed arm. The
            # rounded skin end sits in front, covering the inferred white fill.
            upper.rgba[~full | ~sleeve] = 0
            front_depth = max(1, min(int(upper.z), int(fore.z)))
            fore.depth[:] = front_depth
            upper.depth[:] = min(front_depth+1, 254)
            fore.pivot = [407.0, 508.0]
            fore.meta.update({'joint_cap_inferred_pixels': count, 'joint_round_radius_px': 26})
            upper.meta.update({'joint_cap_inferred_pixels': sleeve_count, 'joint_round_radius_px': 35})
        else:
            distal = _curved_joint_region(full.shape, [658, 457], [
                [[639, 464], [627, 486], [632, 510]],
                [[635, 532], [650, 547], [670, 551]],
            ], [[835, 575], [835, 345], [700, 345], [700, 449]])
            distal = cv2.dilate(distal.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            fore_known = largest(full & distal)
            fore_known |= full & (yy >= 522)
            elbow_outline = ((xx-669)/39)**2 + ((yy-498)/60)**2 <= 1
            fore_known &= (xx >= 669) | elbow_outline
            upper_known = largest(full & ~fore_known)
            upper_known &= yy < 522
            # Dark elbow connector under the complete round gold elbow plate.
            rgb = rgba[..., :3].astype(np.int16)
            rubber = (rgb.max(axis=-1) < 100) & (xx > 590) & (yy > 459)
            upper.rgba, count = _complete_round_joint(rgba, upper_known, (669, 494), 25, rubber,
                                                     bridge_from=(624, 489))
            fore.rgba = _feature_rgba(rgba, fore_known)
            fore.pivot = [669.0, 494.0]
            front_depth = max(1, min(int(upper.z), int(fore.z)))
            fore.depth[:] = front_depth
            upper.depth[:] = min(front_depth+1, 254)
            upper.meta.update({'joint_cap_inferred_pixels': count, 'joint_round_radius_px': 25})
        for p in (upper, fore):
            p.rgba[~largest(p.mask)] = 0
            # Thin prediction spikes are not clothing folds or finger parts.
            cleaned = cv2.morphologyEx(p.mask.astype(np.uint8), cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
            p.rgba[~cleaned] = 0
            p.meta.pop('joint_overlap_px', None)
            p.meta.update({'joint_finish': 'reviewed_curved_material_seam_with_round_underlap',
                           'texture': 'original_visible_pixels_with_bounded_joint_completion',
                           'joint_provenance': 'reviewed_material_seam_and_visible_hinge'})
        if side == 'left':
            # The skin boundary was previously hidden by the cuff's black
            # line. Continue that line around the independent rounded end.
            edge_distance = ndi.distance_transform_edt(fore.mask)
            nearby = np.hypot(xx-407, yy-508) < 38
            outline = fore.mask & nearby & (edge_distance < 2.5)
            weight = np.clip((2.5-edge_distance)/1.5, 0, 1)
            fore.rgba[outline, :3] = np.round(fore.rgba[outline, :3]*(1-weight[outline, None]) +
                                              np.array([23, 23, 25])*weight[outline, None]).astype(np.uint8)
    return layers


def _clean_engineer_face(layers: list[Layer]) -> list[Layer]:
    """Remove a baked hair-shaped cast shadow from the independent face base."""
    by = {p.name: p for p in layers}
    hair = by['hair']
    if 'hair_back' in by:
        # Restore the exact original full hair image, without cutting holes.
        back = by['hair_back']
        choose = back.rgba[..., 3] > hair.rgba[..., 3]
        hair.rgba[choose] = back.rgba[choose]
        hair.meta.pop('partition', None)
        hair.meta.pop('label', None)
        layers = [p for p in layers if p.name != 'hair_back']
    face = by['face']
    rgb = face.rgba[..., :3].astype(np.int16)
    yy, xx = np.indices(face.mask.shape)
    ys, xs = np.where(face.mask)
    clean = face.mask & (rgb[..., 0] > 245) & (rgb[..., 1] > 205) & (rgb[..., 2] > 180)
    clean &= (yy > np.percentile(ys, 72)) & (ndi.distance_transform_edt(face.mask) > 4)
    shadow = face.mask & (rgb[..., 0] > 120) & (rgb[..., 0] < 245) & (rgb[..., 1] < 195)
    shadow &= (rgb[..., 0] > rgb[..., 1]+14) & (rgb[..., 1] > 65)
    if shadow.any() and clean.sum() >= 20:
        material = np.median(face.rgba[clean, :3], axis=0)
        # The independent face base contains no eyes or mouth. A clean skin
        # field removes the entire baked fringe silhouette, including its rim.
        inside = ndi.distance_transform_edt(face.mask)
        repair = face.mask & (rgb[..., 0] > 120) & (rgb[..., 1] > 65)
        weight = np.clip((inside-1.5)/1.5, 0, 1)
        face.rgba[repair, :3] = np.round(face.rgba[repair, :3]*(1-weight[repair, None])+
                                        material*weight[repair, None]).astype(np.uint8)
        face.meta['face_base_cleanup'] = 'hair-shaped brown cast shadow removed; outer contour retained'
    return layers


def apply_reviewed_repairs(name: str, src: np.ndarray, layers: list[Layer],
                           use_lama: bool = True) -> list[Layer]:
    if name in {'025皮卡丘','025皮卡丘小智版'}:
        layers=_repair_pikachu(src,layers,name.endswith('小智版'))
    elif name=='006喷火龙Ymega':
        shadow=ground_shadow(src.shape[:2],(633,922),(270,77),'objects',125)
        layers=[p for p in layers if p.name!='objects']+[shadow]
    elif name=='45-meowa-watermark':
        layers=_animal_sheet(src)
    elif name=='62a3ea887f36c3d5ed44081cf4bdcb1b':
        layers=_repair_doraemon(src,layers)
    elif name=='078f35a9b6e4f0285f2d2b883e52d57d':
        layers=_repair_cat(layers)
    elif name=='sc_gen_7245-meowa-watermark':
        layers=_separate_sheet_tail(src,layers)
    elif name=='301102_malilian_p':
        layers=_separate_fur(layers,use_lama)
    elif name=='sc_gen_498564-meowa-watermark':
        layers=_finish_engineer_arm_joints(layers)
        layers=_clean_engineer_face(layers)
    return [p for p in layers if p.mask.any()]
