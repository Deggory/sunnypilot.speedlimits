# [reviewed-dataset] - START
import json
from test_annotations import reviewed, write, invoke


def make_dataset(root,capsys,count=8):
    rows=[reviewed(root,f'{i+1:08x}--abcdef1234',negative=i==0) for i in range(count)]
    return import_rows(root,capsys,rows)


def import_rows(root,capsys,rows):
    source=write(root,rows)
    code,out=invoke(capsys,'import-annotations',str(source),'--data-root',str(root))
    assert code==0,out.err+out.out
    return root/'annotations/objects.jsonl'


def route_frames(root,route,human,ai=0):
    """human frames confirmed by a person, then ai frames confirmed by an AI reviewer."""
    rows=[]
    for index in range(human+ai):
        row=reviewed(root,route,index=index)
        if index>=human:
            row.update(reviewer='ai/Claude',label_basis='ai_review',reviewer_kind='ai',review_channel='mcp')
        rows.append(row)
    return rows


def split_of(root,capsys,*args):
    code,out=invoke(capsys,'build-splits','--data-root',str(root),*args)
    assert code==0,out.err+out.out
    report=json.loads(out.out)
    return report,json.loads((root/report['split_path']).read_text())


def partition_routes(split,rows):
    routes={}
    for r in rows:
        if r['annotation_id'] in split['assignments']:
            routes.setdefault(split['assignments'][r['annotation_id']],set()).add(r['route_id'][:8])
    return routes


def test_split_defaults_are_deterministic_and_route_disjoint(tmp_path,capsys):
    make_dataset(tmp_path,capsys)
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    report=json.loads(out.out);path=tmp_path/report['split_path'];before=path.read_bytes()
    splits=json.loads(before)
    # Only train, validation and test: every reviewed route is used.
    assert set(splits['assignments'].values())=={'train','validation','test'}
    assert splits['policy']=='train-validation-test-v2' and report['policy']==splits['policy']
    assert set(splits['counts'])=={'train','validation','test'} and sum(splits['counts'].values())==8
    assert splits['human_reviewed_counts']==splits['counts']
    assert sorted(r for routes in splits['routes'].values() for r in routes)==[f'{i+1:08x}--abcdef1234' for i in range(8)]
    assert len(splits['assignments'])==8
    assert splits['leakage_checked'] is True
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    assert (tmp_path/json.loads(out.out)['split_path']).read_bytes()==before
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path),'--splits',str(path))
    assert code==0,out.err+out.out


def test_too_few_independent_routes_fail_without_random_frame_split(tmp_path,capsys):
    make_dataset(tmp_path,capsys,2)
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path))
    assert code!=0 and 'insufficient_independent_groups' in out.out+out.err
    # helpers/3_train.sh reads the number of groups from this text.
    assert ': 2 groups;' in out.out
    assert not (tmp_path/'splits').exists()


def test_three_routes_are_enough_one_per_partition(tmp_path,capsys):
    make_dataset(tmp_path,capsys,3)
    report,split=split_of(tmp_path,capsys)
    assert report['counts']=={'train':1,'validation':1,'test':1}


def test_evaluation_partitions_take_about_15_percent_and_the_rest_is_train(tmp_path,capsys):
    rows=[]
    for i in range(20):
        rows.extend(route_frames(tmp_path,f'{i+1:08x}--abcdef1234',0,2))
    import_rows(tmp_path,capsys,rows)
    report,split=split_of(tmp_path,capsys)
    # 40 rows: about 6 each for validation and test (three routes), the other fourteen routes train.
    assert report['counts']=={'train':28,'validation':6,'test':6}
    assert [len(split['routes'][name]) for name in ('train','validation','test')]==[14,3,3]
    # No frame is a reviewed negative: hard negatives are not required any more.
    assert all(r['detections'] for r in rows)


def test_test_prefers_human_reviewed_groups_that_fit(tmp_path,capsys):
    rows=route_frames(tmp_path,'00000001--abcdef1234',10)
    rows+=route_frames(tmp_path,'00000002--abcdef1234',2,1)
    for i in range(3,11):
        rows+=route_frames(tmp_path,f'{i:08x}--abcdef1234',0,2)
    dataset=import_rows(tmp_path,capsys,rows)
    canonical=[json.loads(s) for s in dataset.read_text().splitlines()]
    for seed in range(4):
        report,split=split_of(tmp_path,capsys,'--seed',str(seed),'--output',str(tmp_path/f'split-{seed}.json'))
        routes=partition_routes(split,canonical)
        # 29 rows, about 4 for test: route 2 (two frames of a person) fits; route 1 (ten) is more than twice that.
        assert '00000002' in routes['test'] and '00000001' not in routes['test'],seed
        assert split['human_reviewed_counts']['test']==2
        assert sum(split['human_reviewed_counts'].values())==12


def test_test_route_pins_a_route_to_test(tmp_path,capsys):
    rows=route_frames(tmp_path,'00000001--abcdef1234',10)
    rows+=route_frames(tmp_path,'00000002--abcdef1234',2,1)
    for i in range(3,11):
        rows+=route_frames(tmp_path,f'{i:08x}--abcdef1234',0,2)
    dataset=import_rows(tmp_path,capsys,rows)
    canonical=[json.loads(s) for s in dataset.read_text().splitlines()]
    report,split=split_of(tmp_path,capsys,'--test-route','00000001--abcdef1234')
    # The pinned route alone is already more than 15%: nothing is added to it.
    assert partition_routes(split,canonical)['test']=={'00000001'}
    assert split['test_routes']==['00000001--abcdef1234'] and report['human_reviewed_counts']['test']==10
    report,split=split_of(tmp_path,capsys,'--test-route','00000005--abcdef1234','--test-route','00000006--abcdef1234')
    routes=partition_routes(split,canonical)
    assert {'00000005','00000006'}<=routes['test'] and report['counts']['test']>=4
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path),'--test-route','00000099--abcdef1234')
    assert code!=0 and 'unknown_test_route' in out.out


def test_hard_negatives_stay_with_their_route(tmp_path,capsys):
    rows=[]
    for i in range(6):
        route=f'{i+1:08x}--abcdef1234'
        rows.append(reviewed(tmp_path,route,index=0))
        rows.append(reviewed(tmp_path,route,index=1,negative=True))
    dataset=import_rows(tmp_path,capsys,rows)
    canonical=[json.loads(s) for s in dataset.read_text().splitlines()]
    report,split=split_of(tmp_path,capsys)
    for route in {r['route_id'] for r in canonical}:
        assert len({split['assignments'][r['annotation_id']] for r in canonical if r['route_id']==route})==1


def test_old_five_partition_split_files_still_validate(tmp_path,capsys):
    dataset=make_dataset(tmp_path,capsys,5)
    rows=[json.loads(s) for s in dataset.read_text().splitlines()]
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path))
    names=['hard-negative','route-held-out','validation','test','train']
    old={'schema_version':1,'dataset_kind':'gold','dataset_sha256':json.loads(out.out)['dataset_sha256'],
         'assignments':{r['annotation_id']:names[i] for i,r in enumerate(sorted(rows,key=lambda r:r['route_id']))}}
    path=tmp_path/'old-split.json';path.write_text(json.dumps(old))
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path),'--splits',str(path))
    assert code==0,out.err+out.out


def test_shared_sites_are_grouped_before_assignment(tmp_path,capsys):
    path=make_dataset(tmp_path,capsys)
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    for r in rows:
        if r['route_id'].startswith(('00000002','00000003')):r['site_group']='same-physical-site'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    code,out=invoke(capsys,'build-splits','--data-root',str(tmp_path))
    assert code==0,out.err+out.out
    result=json.loads((tmp_path/json.loads(out.out)['split_path']).read_text())
    assigned=[result['assignments'][r['annotation_id']] for r in rows if r['site_group']=='same-physical-site']
    assert len(set(assigned))==1
    # A manually modified split must be detected as leakage rather than accepted.
    targets=[r for r in rows if r['site_group']=='same-physical-site']
    result['assignments'][targets[0]['annotation_id']]='train'
    result['assignments'][targets[1]['annotation_id']]='test'
    changed=tmp_path/'leaked.json';changed.write_text(json.dumps(result))
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path),'--splits',str(changed))
    assert code!=0 and 'split_leakage' in out.out
# [reviewed-dataset] - END

# [reviewed-dataset] - START
def test_invalid_split_values_are_reported_without_traceback(tmp_path,capsys):
    dataset=make_dataset(tmp_path,capsys)
    rows=[json.loads(s) for s in dataset.read_text().splitlines()]
    split=tmp_path/'invalid-splits.json'
    split.write_text(json.dumps({'schema_version':1,'assignments':{r['annotation_id']:[] for r in rows}}))
    code,out=invoke(capsys,'validate-dataset','--data-root',str(tmp_path),'--splits',str(split))
    assert code!=0 and 'invalid_split_membership' in out.out
# [reviewed-dataset] - END


# [place-split] - START
def located_frames(root,route,cells,human=True):
    """One frame per area of about 300 m along the same road, each with its own image and GPS position."""
    import hashlib
    from test_annotations import png
    rows=[]
    for index,cell in enumerate(cells):
        row=reviewed(root,route,index=index)
        image=root/row['image_path']
        png(image,(int(route[:8],16)*40+index*7+1)%256)
        row['frame_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
        row['gps']={'latitude':47.0+cell*0.0027+0.001,'longitude':8.0+0.002,'speed_mps':20.0,'bearing_deg':0.0}
        row['country']='CH'
        if not human:
            row.update(reviewer='ai/Claude',label_basis='ai_review',reviewer_kind='ai',review_channel='mcp')
        rows.append(row)
    return rows


def test_split_by_place_keeps_the_test_areas_and_a_buffer_out_of_training(tmp_path,capsys):
    from nnslr_tools.annotations import load_jsonl
    from nnslr_tools.splits import place_of
    road=list(range(20))
    rows=(located_frames(tmp_path,'00000001--abcdef1234',road)
          +located_frames(tmp_path,'00000002--abcdef1234',road,human=False)
          +located_frames(tmp_path,'00000003--abcdef1234',road,human=False))
    dataset=import_rows(tmp_path,capsys,rows)
    report,split=split_of(tmp_path,capsys,'--test-route','00000001--abcdef1234')
    assert split['policy']=='place-v3' and split['grouping'][0]=='place'
    canonical=load_jsonl(dataset)
    where={}
    for row in canonical:
        where.setdefault(place_of(row),set()).add(split['assignments'][row['annotation_id']])
    # One partition per area, whatever route drove through it.
    assert all(len(names)==1 for names in where.values())
    areas={name:{place for place,names in where.items() if name in names} for name in ('train','validation','test','buffer')}
    assert areas['test'] and areas['buffer'] and areas['train'] and areas['validation']
    # Nothing next to a test area is trained on.
    for a,b in areas['test']:
        for da in (-1,0,1):
            for db in (-1,0,1):
                assert (a+da,b+db) not in areas['train']
    assert split['counts']['test']<=0.3*len(canonical)
# [place-split] - END
