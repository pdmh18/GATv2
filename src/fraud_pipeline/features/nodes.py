"""Exact node history at a cutoff, derived from the sparse event timeline."""
NODE_DYNAMIC_NAMES = ('log_incoming_settled_count', 'log_outgoing_settled_count',
                      'log_unique_incoming_partners', 'log_unique_outgoing_partners',
                      'log_incoming_median_amount', 'log_outgoing_median_amount',
                      'has_incoming_history', 'has_outgoing_history')


def dynamic_features(arrays, at, end, node_count, seconds):
    import numpy as np
    times = arrays['graph_timestamps_us']
    start = int(np.searchsorted(times[:end], at - seconds * 1000000, side='left'))
    settled = arrays['graph_kind'][start:end] == 1
    edges = arrays['graph_edge_index'][:, start:end][:, settled]
    amounts = np.expm1(np.asarray(arrays['graph_log_amount'][start:end][settled], dtype='float64'))
    result = np.zeros((node_count, len(NODE_DYNAMIC_NAMES)), dtype='float32')
    for ids, count_column, degree_column, amount_column, flag_column, partner_ids in [
        (edges[1], 0, 2, 4, 6, edges[0]), (edges[0], 1, 3, 5, 7, edges[1])]:
        if not len(ids):
            continue
        order = np.argsort(ids, kind='stable')
        sorted_ids = ids[order]
        nodes, first, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
        result[nodes, count_column] = np.log1p(counts)
        result[nodes, flag_column] = 1
        for node, offset, count in zip(nodes, first, counts):
            selected = order[offset:offset + count]
            result[node, degree_column] = np.log1p(len(np.unique(partner_ids[selected])))
            result[node, amount_column] = np.log1p(np.median(amounts[selected]))
    return result
