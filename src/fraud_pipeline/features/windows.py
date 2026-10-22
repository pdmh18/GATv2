"""Explicit fixed-duration window profiles. One month means 30 days, not calendar month."""
import copy
import re

UNITS={'s':1,'m':60,'h':3600,'d':86400,'w':604800,'mo':2592000}


def parse_window(value):
    if not isinstance(value,str):raise ValueError('Window must be a duration string')
    match=re.fullmatch(r'([1-9][0-9]*)(mo|s|m|h|d|w)',value.strip())
    if not match:raise ValueError('Window must be a positive duration, e.g. 1d, 1w, 1mo (30 days), 5m')
    return int(match[1])*UNITS[match[2]]


def window_profile(config, *, window=None, amount_window=None, median_window=None, graph_window=None, node_window=None):
    result=copy.deepcopy(config)
    overrides={}
    if window is not None:
        value=parse_window(window)
        overrides.update(mean_window_seconds=value,amount_window_seconds=value,graph_window_seconds=value,node_history_window_seconds=value)
    for value,key in ((amount_window,'mean_window_seconds'),(median_window,'amount_window_seconds'),
                      (graph_window,'graph_window_seconds'),(node_window,'node_history_window_seconds')):
        if value is not None:overrides[key]=parse_window(value)
    if not overrides:raise ValueError('Specify --window or a specific window override')
    result['features'].update(overrides)
    # Canonical seconds ensure aliases such as 1w/7d map to the same profile.
    f=result['features']
    tag=f'mean-{f.get("mean_window_seconds",0)}s_median-{f["amount_window_seconds"]}s_graph-{f["graph_window_seconds"]}s_node-{f["node_history_window_seconds"]}s'
    root=result['dataset']['run_root'].rstrip('/')+f'/seed-{result["dataset"]["seed"]}'
    result['dataset']['features_output']=root+'/features-v3/'+tag
    result['training']['output']=root+'/training/gatv2-v3/'+tag
    result['evaluation']['output']=root+'/evaluation/gatv2-v3/'+tag
    result['feature_eda']['output']=root+'/eda/features-v3/'+tag
    result['window_profile']={'id':tag,'duration_policy':'Fixed duration; 1mo=30 days. Request-count windows remain independent.'}
    return result
