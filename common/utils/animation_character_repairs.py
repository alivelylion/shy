"""Reviewed contour repairs for the four cartoon examples.

All fills are bounded completions of observed parts. No unseen limbs are added.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from pathlib import Path
import json

from utils.animation_parts import Layer
from utils.animation_integrity import polygon
from utils.animation_standard import ellipse, curved_region, reset_records


def ink_path(shape: tuple[int,int], commands: list[tuple], colour: list[int],
             outline: list[int] | None = None, width: float = 3) -> np.ndarray:
    import aggdraw
    image=Image.new('RGBA',(shape[1],shape[0]))
    drawing=aggdraw.Draw(image);path=aggdraw.Path()
    for command in commands:
        tag,*values=command
        if tag=='M':path.moveto(*values)
        elif tag=='C':path.curveto(*values)
        elif tag=='L':path.lineto(*values)
        elif tag=='Z':path.close()
        else:raise ValueError(tag)
    drawing.path(path,aggdraw.Brush(tuple(colour)),aggdraw.Pen(tuple(outline),width) if outline else None)
    drawing.flush()
    return np.array(image)


def diffuse_material(src: np.ndarray, support: np.ndarray, trusted: np.ndarray,
                     guides: list[tuple[np.ndarray,list[float]]] | None = None) -> np.ndarray:
    """Solve colour diffusion inside one silhouette; never copy rectangular tiles."""
    rgb=src[...,:3].astype(float).copy()
    fixed=trusted.copy()
    for mask,colour in guides or []:
        selected=mask&support&~trusted
        rgb[selected]=colour;fixed|=selected
    labels,count=ndi.label(support)
    for component in range(1,count+1):
        island=labels==component
        if not (island&fixed).any():
            if not fixed.any():raise ValueError('An observed material sample is required')
            rgb[island]=np.median(rgb[fixed],axis=0);fixed|=island
    unknown=support&~fixed
    ys,xs=np.where(unknown);count=len(xs)
    if not count:return np.round(rgb).astype(np.uint8)
    if not fixed.any():raise ValueError('An observed material sample is required')
    ids=np.full(support.shape,-1,np.int32);ids[ys,xs]=np.arange(count)
    diagonal=np.zeros(count,float);rhs=np.zeros((count,3),float)
    row=[];col=[];weight=[]
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx,ny=xs+dx,ys+dy
        valid=(nx>=0)&(ny>=0)&(nx<support.shape[1])&(ny<support.shape[0])
        chosen=np.flatnonzero(valid);chosen=chosen[support[ny[chosen],nx[chosen]]]
        diagonal[chosen]+=1
        neighbours=ids[ny[chosen],nx[chosen]]
        missing=neighbours>=0
        row.extend(chosen[missing]);col.extend(neighbours[missing]);weight.extend([-1.]*int(missing.sum()))
        fixed_ids=chosen[~missing]
        rhs[fixed_ids]+=rgb[ny[fixed_ids],nx[fixed_ids]]
    row.extend(range(count));col.extend(range(count));weight.extend(diagonal)
    matrix=coo_matrix((weight,(row,col)),shape=(count,count)).tocsr()
    rgb[ys,xs]=spsolve(matrix,rhs)
    return np.round(rgb).clip(0,255).astype(np.uint8)


def rebuilt_charizard_tail(src: np.ndarray, layers: list[Layer]) -> tuple[list[Layer],dict[str,np.ndarray]]:
    shape=src.shape[:2];yy,xx=np.indices(shape)
    commands=[('M',156,342),('C',150,391,133,457,140,505),
        ('C',148,603,188,682,255,738),('C',305,782,365,819,431,832),
        ('C',481,838,534,807,557,774),('C',574,751,568,730,548,717),
        ('C',513,694,476,673,450,651),('L',420,626),
        ('C',423,649,435,671,432,685),('C',411,707,358,705,315,688),
        ('C',267,671,238,625,216,569),('C',192,508,177,448,173,392),
        ('L',169,349),('C',166,341,160,338,156,342),('Z',)]
    rgba=ink_path(shape,commands,[249,165,92],[106,65,42],3)
    support=rgba[...,3]>10
    wing=polygon(shape,[[522,125],[463,244],[465,347],[478,415],[513,450],[598,483],
        [537,572],[503,613],[455,542],[422,445],[446,596],[435,646],[417,697],
        [404,742],[400,792],[397,855],[370,835],[334,794],[291,725],[268,654],
        [241,573],[233,513],[252,448],[279,386],[327,334],[400,277],[470,177]])
    leg=polygon(shape,[[491,697],[545,682],[584,694],[609,728],[621,779],[607,825],
        [579,858],[538,869],[502,858],[478,835],[472,797],[475,760],[483,725]])
    arm=polygon(shape,[[397,440],[426,498],[493,516],[506,570],[524,594],[553,573],
        [562,597],[558,638],[557,671],[552,704],[535,720],[501,714],[494,679],
        [479,654],[447,619],[419,589],[405,543]])
    torso=polygon(shape,[[594,530],[686,487],[742,596],[779,752],[739,829],[689,850],[635,848],[598,793],[611,701]])
    known=support&(src[...,3]>10)&~wing&~leg&~arm&~torso&(yy>=355)
    rgb=src[...,:3].astype(int)
    clearance=ndi.distance_transform_edt(~(wing|leg|arm|torso))
    trusted=known&(clearance>14)&(rgb[...,0]>155)&(rgb[...,1]>90)&(rgb[...,2]<190)
    trusted &= cv2.erode(support.astype(np.uint8),np.ones((5,5),np.uint8))>0
    # The small right-hand islands contain contact shadows from the wing and
    # thigh. They are colour references, not patches to paste into a full tail.
    trusted &= xx<270
    orange=src[known&(rgb[...,0]-rgb[...,1]>45)&(rgb[...,1]>100)&(rgb[...,1]<205),:3]
    cream=src[known&(rgb[...,0]>185)&(rgb[...,1]>170)&(rgb[...,2]>100),:3]
    orange_colour=np.median(orange,axis=0).tolist() if len(orange) else [249,165,92]
    cream_colour=np.median(cream,axis=0).tolist() if len(cream) else [242,216,149]
    def guide(points: list[list[int]],width: int) -> np.ndarray:
        mask=np.zeros(shape,np.uint8)
        cv2.polylines(mask,[np.asarray(points,np.int32)],False,1,width,cv2.LINE_AA)
        return (mask>0)&~trusted
    guides=[(guide([[187,608],[208,658],[244,708],[292,754],[348,787],[402,809],[455,811],[502,784]],5),cream_colour),
            (guide([[169,500],[197,605],[239,676],[299,733],[375,766],[449,752],[532,748]],7),orange_colour)]
    material=diffuse_material(src,support,trusted,guides)
    rgba[support,:3]=material[support]
    # A single outer rim, including the previously hidden wing crossing.
    rim=support&(ndi.distance_transform_edt(support)<=2)
    rgba[rim,:3]=[106,65,42]
    tail=Layer('tail',rgba,np.full(shape,200,np.uint8),'reviewed:source_tail',
        parent='topwear',pivot=[538,756],role='animal_tail',completed=support&~known,
        meta={'label':'尾巴','completion':'continuous source-colour diffusion across the wing occlusion',
              'provenance':'observed tail with a single rounded attachment; no pasted body or leg pixels'})
    fire=next(p for p in layers if p.name=='earwear')
    fire_known=(src[...,3]>10)&(yy<355)&(xx<245)
    fire.rgba=src.copy();fire.rgba[~fire_known]=0;fire.parent='tail';fire.pivot=[156,343]
    result=[p for p in layers if p.name!='tail']+[tail]
    return result,{'tail':known,'earwear':fire_known}


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))>0


def _largest(mask: np.ndarray) -> np.ndarray:
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return labels==(1+int(stats[1:,4].argmax())) if count>1 else mask


def _over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    alpha=top[...,3:4].astype(float)/255
    base_alpha=bottom[...,3:4].astype(float)/255
    output_alpha=alpha+base_alpha*(1-alpha)
    numerator=top[...,:3]*alpha+bottom[...,:3]*base_alpha*(1-alpha)
    result=np.zeros_like(bottom)
    result[...,:3]=np.round(np.divide(numerator,output_alpha,
        out=np.zeros_like(numerator),where=output_alpha>0)).astype(np.uint8)
    result[...,3]=np.round(output_alpha[...,0]*255).astype(np.uint8)
    return result


def _ellipse_ink(shape: tuple[int,int], values: list[float], colour: list[int],
                 outline: list[int] | None=None, width: float=3) -> np.ndarray:
    x,y,rx,ry=values;k=.55228475
    return ink_path(shape,[('M',x+rx,y),('C',x+rx,y+k*ry,x+k*rx,y+ry,x,y+ry),
        ('C',x-k*rx,y+ry,x-rx,y+k*ry,x-rx,y),('C',x-rx,y-k*ry,x-k*rx,y-ry,x,y-ry),
        ('C',x+k*rx,y-ry,x+rx,y-k*ry,x+rx,y),('Z',)],colour,outline,width)


def _reviewed_layer(name: str, rgba: np.ndarray, role: str, depth: int,
                    parent: str | None, pivot: list[float], label: str) -> Layer:
    return Layer(name,rgba,np.full(rgba.shape[:2],depth,np.uint8),'reviewed:source_character',
        role=role,parent=parent,pivot=pivot,meta={'label':label,
        'completion':'reviewed continuous contour and observed material; hidden attachments estimated'})


def _feature(src: np.ndarray, boxes: list[list[int]], threshold: int) -> np.ndarray:
    """Extract whole observed facial marks, including their white highlights."""
    result=np.zeros(src.shape[:2],bool)
    for x1,y1,x2,y2 in boxes:
        rgb=src[y1:y2,x1:x2,:3]
        dark=(rgb[...,1]<threshold)&(rgb[...,0]<215)
        count,labels,stats,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8),8)
        keep=np.zeros(count,bool);keep[1:]=stats[1:,4]>=3
        mark=keep[labels]
        contours,_=cv2.findContours(mark.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filled=np.zeros(mark.shape,np.uint8)
        if contours:cv2.fillConvexPoly(filled,cv2.convexHull(np.concatenate(contours)),1)
        mark=filled>0
        # A narrow source-colour edge keeps the original low-resolution artwork.
        result[y1:y2,x1:x2]|=_dilate(mark,1)
    return result


def rebuild_pikachu(src: np.ndarray, layers: list[Layer], info: dict,
                    name: str) -> tuple[list[Layer],dict]:
    """Repartition observed short limbs; never segment cartoon elbows or knees."""
    hints=Path(__file__).resolve().parents[2]/'workspace/animation_integrity_hints.json'
    layout=json.loads(hints.read_text(encoding='utf-8'))[name]['source_instances'][0]
    shape=src.shape[:2];yy,xx=np.indices(shape);rgb=src[...,:3].astype(int)
    ash='小智' in name;prefix=layout['name'];body_name=prefix+'_body';head_name=prefix+'_head'
    old={p.name:p for p in layers}
    yellow=(rgb[...,0]>150)&(rgb[...,1]>105)&(rgb[...,0]>rgb[...,2]+45)
    visible=(src[...,3]>50)&_dilate(yellow,4)&~((rgb[...,1]>rgb[...,0]+12)&(rgb[...,0]<180))
    body_points=([[457,459],[510,434],[618,424],[718,440],[777,478],[806,560],
        [824,657],[840,766],[831,844],[797,885],[735,902],[666,885],[609,878],
        [547,891],[483,886],[436,856],[417,815],[421,750],[435,679],[446,583],[457,459]] if ash else
        [[266,645],[294,574],[350,516],[423,470],[502,439],[577,435],[632,457],
         [664,502],[662,567],[641,639],[598,701],[534,738],[457,763],[383,761],[313,739],[273,699],[266,645]])
    head_points=([[460,255],[488,221],[588,203],[692,207],[775,236],[815,292],
        [829,360],[815,421],[779,465],[722,490],[643,497],[560,486],[495,456],
        [461,407],[445,346],[449,293],[460,255]] if ash else
        [[543,477],[556,422],[602,377],[664,353],[745,354],[836,370],[908,405],
         [950,460],[974,529],[977,605],[957,673],[920,715],[852,743],[771,751],
         [695,740],[626,720],[573,677],[546,616],[535,548],[543,477]])
    supports={body_name:curved_region(shape,body_points),head_name:curved_region(shape,head_points)}
    regions={d['name']:polygon(shape,d['polygon']) for d in layout['regions'] if 'shadow' not in d['name']}
    regions.update({d['name']:polygon(shape,d['polygon']) for d in layout['chains']})
    limbs={d['name'] for d in layout['chains']}
    ears={n for n in regions if '_ear_' in n};hat=prefix+'_hat'
    # Source masks preserve toes and zigzag tail corners. Only the obscured root
    # receives a smooth continuation, instead of enlarging the entire limb.
    masks={n:_largest(region&visible) for n,region in regions.items() if n not in supports}
    if ash:
        # Dark ear tips are intentional artwork, and need explicit inclusion.
        for n in ears:
            masks[n]=_largest(regions[n]&(src[...,3]>100))
        hat_region=polygon(shape,[[460,196],[466,95],[482,55],[519,27],[600,9],[672,0],
            [717,18],[770,31],[813,63],[845,109],[857,215],[883,249],[906,283],[903,293],
            [849,274],[815,260],[756,250],[666,247],[594,236],[543,222],[514,211]])
        masks[hat]=_largest(hat_region&(src[...,3]>30))
        for n in ears:
            masks[n]&=~_dilate(masks[hat],2)
            masks[n]&=~((rgb[...,0]>150)&(rgb[...,1]<130)&(rgb[...,2]<120))
            if n.endswith('right'):masks[n]&=~((xx<827)&(yy>345))
    else:
        for n in ears:
            masks[n]=_largest(regions[n]&(src[...,3]>100))
    if ash:
        eye_boxes=[[540,267,626,339],[729,277,802,349]]
        nose_boxes=[[686,336,720,363]];mouth_boxes=[[630,368,744,403]]
        caps={'arm_left':[492,532,28,31],'arm_right':[755,535,25,30],
              'leg_left':[483,888,32,25],'leg_right':[792,886,30,23]}
    else:
        eye_boxes=[[692,504,783,608],[888,517,958,610]]
        nose_boxes=[[848,598,884,635]];mouth_boxes=[[772,636,895,733]]
        caps={'front_near':[608,748,54,47],'front_far':[851,762,30,29],
              'hind_near':[295,741,29,36],'hind_far':[504,768,22,26]}
    feature_masks={'eyes':_feature(src,eye_boxes,155),'nose':_feature(src,nose_boxes,190),
                   'mouth':_feature(src,mouth_boxes,175)}
    if not ash:
        # The pink tongue is much brighter than the mouth outline and must be
        # included in both the feature and the face-cleaning region.
        feature_masks['mouth']=polygon(shape,[[779,646],[804,654],[833,653],[856,644],
            [879,658],[862,680],[845,707],[827,724],[804,718],[788,699]])
    limb_outlines=({
        'arm_left':[[470,505],[504,500],[538,520],[564,555],[588,601],[614,646],
            [632,681],[635,703],[620,722],[601,721],[564,698],[532,671],[500,634],
            [473,590],[455,547],[451,521],[470,505]],
        'arm_right':[[740,506],[769,509],[785,532],[782,569],[765,613],[739,656],
            [711,689],[686,709],[671,706],[670,681],[684,643],[690,600],[699,557],[716,520],[740,506]],
        'leg_left':[[462,864],[490,862],[513,878],[508,896],[475,924],[434,951],
            [401,958],[402,939],[422,911],[446,880],[462,864]],
        'leg_right':[[770,866],[793,866],[816,886],[847,907],[859,918],[845,925],
            [817,919],[785,905],[765,884],[770,866]],
    } if ash else {
        'front_near':[[566,702],[600,702],[640,722],[667,751],[666,800],[657,850],
            [650,877],[590,878],[575,841],[560,783],[552,733],[566,702]],
        'front_far':[[829,735],[854,733],[875,747],[882,780],[888,822],[881,835],
            [843,832],[828,796],[817,755],[829,735]],
        'hind_near':[[280,712],[308,718],[324,746],[307,790],[283,831],
            [269,848],[242,847],[242,815],[253,778],[263,740],[280,712]],
        'hind_far':[[487,738],[509,740],[526,767],[542,800],[545,815],[530,817],
            [513,803],[491,775],[481,752],[487,738]],
    })
    result=[]
    for n in sorted(limbs|ears|{prefix+'_tail'}|({hat} if ash else set())):
        known=masks[n].copy();support=known.copy();parent=head_name if n in ears or n==hat else body_name
        pivot=old[n].pivot if n in old else None
        if n in limbs:
            key=n[len(prefix)+1:];cap=caps[key]
            safe_y=({'arm_left':553,'arm_right':555,'leg_left':887,'leg_right':887} if ash else
                    {'front_near':731,'front_far':749,'hind_near':762,'hind_far':735})[key]
            known=regions[n]&visible&(yy>=safe_y)
            known=cv2.morphologyEx(known.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
            support=curved_region(shape,limb_outlines[key]);known&=support
            pivot=cap[:2]
            # Interior at the attachment excludes the old torso/limb cut line.
            root=ellipse(shape,[*cap[:2],cap[2]*.8,cap[3]*.8])
            trusted=known&~root
            trusted&=yy>safe_y+8
            if not ash:trusted&=yellow|((rgb[...,1]>70)&(rgb[...,0]>100))
            yellow_samples=trusted&yellow
            base=np.median(src[yellow_samples,:3],axis=0) if yellow_samples.any() else [250,215,55]
            blank=src.copy();blank[support&~known,:3]=base
            material=diffuse_material(blank,support,trusted,[(ellipse(shape,[*cap[:2],3,3]),np.asarray(base).tolist())])
            rgba=np.zeros_like(src);rgba[support,:3]=material[support];rgba[support,3]=255
            # The observed lower limb remains intact, including toes. The root
            # is continuous with it, rather than a polygon pasted from the torso.
            role='arm_whole' if ('arm_' in n or 'front_' in n) else 'leg_whole'
            depth=(150 if 'hind_' in n else 80) if not ash else (150 if 'leg_' in n else 40)
        else:
            rgba=src.copy();rgba[~known]=0;role='animal_tail' if n.endswith('_tail') else 'source_region'
            depth=160 if role=='animal_tail' else (10 if n==hat else 110)
            if n in ears:
                if ash:
                    cap=[512,226,24,21] if n.endswith('left') else [835,309,21,30]
                    if n.endswith('right'):
                        support=curved_region(shape,[[826,279],[847,289],[857,333],[868,394],
                            [869,414],[860,459],[857,477],[835,448],[819,405],[813,359],[817,313],[826,279]])
                    else:
                        support=ink_path(shape,[('M',191,249),('C',239,203,357,170,467,186),
                            ('C',511,196,545,213,546,231),('C',468,269,313,282,191,249),('Z',)],
                            [252,222,49])[...,3]>10
                else:
                    cap=[624,411,37,25] if n.endswith('near') else [914,401,30,39]
                    commands=([('M',354,279),('C',433,263,555,324,649,393),
                        ('C',663,409,670,424,659,432),('C',580,444,445,374,354,279),('Z',)]
                        if n.endswith('near') else [('M',806,88),('C',844,98,890,161,918,230),
                        ('C',949,303,957,379,950,439),('C',915,431,878,385,854,318),
                        ('C',829,247,810,160,806,88),('Z',)])
                    support=ink_path(shape,commands,[252,222,49])[...,3]>10
                    known&=yy<(403 if n.endswith('near') else 381)
                support=_largest(support);known&=support
                root=ellipse(shape,cap)
                pivot=cap[:2]
                trusted=known&~_dilate(root,3)
                material=diffuse_material(src,support,trusted,[(ellipse(shape,[*cap[:2],3,3]),[252,222,49])])
                rgba=np.zeros_like(src);rgba[support,:3]=material[support];rgba[support,3]=255
        if pivot is None:
            y,x=np.where(support);pivot=[float(np.median(x)),float(np.median(y))]
        label=next((d.get('label') for d in layout['regions'] if d['name']==n),None) or ('完整手臂' if role=='arm_whole' else '完整腿')
        p=_reviewed_layer(n,rgba,role,depth,parent,pivot,label);p.completed=support&~known
        if n in limbs:
            p.visibility_mask=((src[...,3]>50).astype(np.uint8)*255)
        result.append(p)
    for n,role,depth,pivot,label in [(body_name,'animal_body',120,[624,650] if ash else [471,606],'躯干'),
                  (head_name,'animal_head',60,[640,470] if ash else [702,673],'头部')]:
        support=supports[n]
        hidden=np.zeros(shape,bool)
        relevant=(limbs|ears|({hat} if ash else set())) if n==body_name else (ears|({hat} if ash else set()))
        for part in relevant:hidden|=_dilate(masks[part],9)
        if n==body_name:hidden|=_dilate(supports[head_name],8)
        else:
            for mask in feature_masks.values():hidden|=_dilate(mask,9)
        trusted=support&(src[...,3]>100)&~hidden&regions[n]
        if n==head_name:
            if not ash:trusted&=~((xx<655)&(yy<515))
        elif ash:
            # Interior arm outlines extend beyond the old polygon prompts.
            for key in ('arm_left','arm_right'):
                hidden|=_dilate(curved_region(shape,limb_outlines[key]),18)
            for p in result:
                if p.name in limbs:hidden|=_dilate(p.mask,12)
            trusted&=~hidden
            lines=np.zeros(shape,np.uint8)
            for points in ([[569,510],[593,559],[612,613],[636,680]],
                           [[691,535],[687,592],[685,652],[691,686]]):
                cv2.polylines(lines,[np.array(points,np.int32)],False,1,30)
            trusted&=~(lines>0)
            trusted&=~polygon(shape,[[751,573],[796,575],[852,769],[821,793],
                                     [780,703],[756,644],[751,573]])
            trusted&=yy>=500
        if n==head_name:
            # Remove the complete feature boxes, including pale antialiasing.
            for x1,y1,x2,y2 in eye_boxes+nose_boxes+mouth_boxes:
                trusted[max(0,y1-5):y2+5,max(0,x1-5):x2+5]=False
        else:
            trusted&=(rgb[...,0]>70)&(rgb[...,1]>40)
            if ash:
                trusted&=rgb[...,1]>120
                # The outer arm outline had been baked into the source torso.
                # Its antialias pixels can be brighter than a simple black
                # threshold, so upper-torso samples must be yellow material.
                trusted&=(yy>=800)|((rgb[...,0]>200)&(rgb[...,1]>170)&(rgb[...,2]<150))
            else:
                stripes=polygon(shape,[[373,477],[460,431],[600,430],[598,568],[429,588],[373,477]])
                trusted&=yellow|stripes
        # Trust only interior material beside newly created silhouettes.
        trusted &= ndi.distance_transform_edt(support)>4
        yellow_trusted=trusted&yellow
        base=np.median(src[yellow_trusted,:3],axis=0).tolist()
        # Colour guides inside newly covered attachments prevent dark outline
        # pixels at the cut from flooding a whole completed head or torso.
        guide=(support&~_dilate(trusted,12))
        material=diffuse_material(src,support,trusted,[(guide,base)])
        rgba=np.zeros_like(src);rgba[support,:3]=material[support];rgba[support,3]=255
        p=_reviewed_layer(n,rgba,role,depth,body_name if n==head_name else None,pivot,label)
        p.completed=support&~trusted
        p.visibility_mask=((src[...,3]>50).astype(np.uint8)*255)
        result.append(p)
    for n,mask in feature_masks.items():
        rgba=src.copy();rgba[~mask]=0
        result.append(_reviewed_layer(n,rgba,'face_feature',20,head_name,
                      [float(np.where(mask)[1].mean()),float(np.where(mask)[0].mean())],
                      {'eyes':'眼睛','nose':'鼻子','mouth':'嘴巴'}[n]))
    shadow=old[prefix+'_shadow'];shadow.depth[:]=250;result.append(shadow)
    return result,reset_records(info,[{'name':prefix,'kind':'biped' if ash else 'quadruped',
             'parts':[p.name for p in result if 'shadow' not in p.name],'inferred_hidden_parts':[]}])


def rebuild_doraemon(src: np.ndarray, layers: list[Layer], info: dict) -> tuple[list[Layer],dict]:
    """Trace the observed flat-colour limbs without collar or belly remnants."""
    shape=src.shape[:2];blue=[8,157,226];white=[252,253,250];ink=[23,29,29]
    old={p.name:p for p in layers};result=[]
    def add(n: str, commands: list[tuple], colour: list[int], role: str,
            depth: int,pivot: list[float],label: str) -> Layer:
        rgba=ink_path(shape,commands,colour,ink,3)
        p=_reviewed_layer(n,rgba,role,depth,'doraemon_body',pivot,label);result.append(p);return p
    body=add('doraemon_body',[('M',310,526),('C',372,514,584,522,649,545),
        ('C',668,580,671,710,657,786),('C',645,824,624,848,582,853),
        ('C',491,862,374,848,305,817),('C',278,795,267,755,268,690),
        ('C',267,616,276,557,310,526),('Z',)],blue,'animal_body',140,[468,668],'躯干与腹部')
    body.parent=None
    bib=_ellipse_ink(shape,[435,662,160,154],white,ink,3)
    body.rgba=_over(body.rgba,bib)
    pocket=ink_path(shape,[('M',309,674),('C',354,670,482,670,556,674),
        ('C',552,738,494,782,429,784),('C',359,784,314,742,309,674),('Z',)],white,ink,3)
    body.rgba=_over(body.rgba,pocket)
    ygrid=np.indices(shape)[0]
    body.rgba[(ygrid>810)&body.mask,:3]=blue
    # Legs have rounded blue roots and their own white feet. There is no belly
    # material in either complete limb PNG.
    left=add('doraemon_leg_left',[('M',276,810),('C',295,782,374,781,403,803),
        ('C',423,825,433,857,433,897),('C',409,913,324,918,292,894),
        ('C',280,871,274,837,276,810),('Z',)],blue,'leg_whole',100,[345,824],'画面左完整腿')
    left.rgba=_over(left.rgba,ink_path(shape,[('M',292,884),('C',267,890,244,906,242,927),
        ('C',236,959,274,969,330,970),('C',375,973,419,969,439,954),
        ('C',449,935,443,912,432,899),('C',397,905,332,907,292,884),('Z',)],white,ink,3))
    right=add('doraemon_leg_right',[('M',428,824),('C',466,788,585,792,630,815),
        ('C',657,833,649,880,630,913),('C',587,927,469,922,437,908),
        ('C',435,880,430,852,428,824),('Z',)],blue,'leg_whole',90,[549,839],'画面右完整腿')
    right.rgba=_over(right.rgba,ink_path(shape,[('M',437,903),('C',416,919,403,935,406,956),
        ('C',408,980,445,991,505,994),('C',562,998,620,989,633,971),
        ('C',647,951,641,923,629,905),('C',575,915,490,913,437,903),('Z',)],white,ink,3))
    for p in (left,right):
        # The body hides the root, while the original leg division remains
        # below its lower rim. Full PNGs keep the overlap for motion.
        p.depth[:]=160
    far=add('doraemon_arm_far',[('M',640,558),('C',659,558,667,583,666,622),('L',665,668),
        ('C',657,692,632,704,594,703),('C',530,703,478,682,429,659),
        ('C',396,639,374,615,357,586),('C',342,570,338,550,351,539),
        ('C',368,525,390,541,409,559),('C',469,570,546,575,610,573),
        ('C',625,569,632,558,640,558),('Z',)],
        blue,'arm_whole',60,[643,584],'远侧完整手臂')
    far.rgba=_over(far.rgba,_ellipse_ink(shape,[365,565,51,46],white,ink,3))
    near=add('doraemon_arm_near',[('M',299,522),('C',267,533,246,566,240,606),
        ('C',234,637,243,659,265,670),('C',295,686,348,687,389,675),
        ('C',420,665,445,647,468,633),('L',486,601),
        ('C',456,582,404,569,366,560),('C',338,555,321,540,299,522),('Z',)],
        blue,'arm_whole',40,[286,556],'近侧完整手臂')
    near.rgba=_over(near.rgba,_ellipse_ink(shape,[468,590,56,57],white,ink,3))
    # Reuse the source-derived head and expression. Only remove blue sleeve /
    # collar remnants from limbs; no new expression or style is generated.
    for n,depth in [('doraemon_head',80),('doraemon_collar',70),('doraemon_tail',160),
                    ('eyes',20),('nose',15),('doraemon_face_lines',25)]:
        p=old[n];p.depth[:]=depth
        if n in {'eyes','nose','doraemon_face_lines'}:p.parent='doraemon_head'
        result.append(p)
    head=next(p for p in result if p.name=='doraemon_head')
    head.rgba=ink_path(shape,[('M',420,53),('C',525,27,664,66,738,126),
        ('C',817,191,830,324,797,417),('C',772,487,718,530,655,547),
        ('C',551,559,387,539,299,503),('C',235,476,206,397,211,317),
        ('C',214,208,297,102,420,53),('Z',)],blue,ink,3)
    face=ink_path(shape,[('M',360,128),('C',309,119,257,165,229,222),
        ('C',210,266,204,328,218,385),('C',231,447,258,482,297,501),
        ('C',386,533,504,553,613,545),('C',687,509,731,444,726,364),
        ('C',723,280,663,213,597,192),('C',552,178,498,142,450,131),
        ('C',415,123,386,122,360,128),('Z',)],white,ink,3)
    head.rgba=_over(head.rgba,face);head.visibility_mask=None
    return result,reset_records(info,[{'name':'doraemon','kind':'biped',
             'parts':[p.name for p in result],'inferred_hidden_parts':[]}])
