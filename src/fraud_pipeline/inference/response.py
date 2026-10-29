"""Canonical model response v1 shared by CLI, Kafka and any future HTTP adapter."""
from math import isfinite

RESPONSE_FIELDS = ('transaction_id', 'risk_score', 'predicted_label', 'threshold',
                   'model_version', 'feature_version')
KEY_FIELDS = ('transaction_id', 'model_version', 'feature_version')
STRING_FIELDS = ('transaction_id', 'model_version', 'feature_version')


def build_response(*, transaction_id, risk_score, threshold, model_version, feature_version):
    """Build and validate the exact payload consumed by the backend."""
    score, cutoff = float(risk_score), float(threshold)
    response = {
        'transaction_id': transaction_id,
        'risk_score': score,
        'predicted_label': int(score >= cutoff),
        'threshold': cutoff,
        'model_version': model_version,
        'feature_version': feature_version,
    }
    return validate_response(response)


def validate_response(response):
    """Reject drift from specs/model_response.v1.json before it reaches the backend."""
    if not isinstance(response, dict):
        raise ValueError('Model response must be an object')
    if set(response) != set(RESPONSE_FIELDS):
        raise ValueError(f'Model response fields must be exactly {RESPONSE_FIELDS}')
    for name in STRING_FIELDS:
        value = response[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Model response {name} must be a nonempty string')
    for name in ('risk_score', 'threshold'):
        value = response[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f'Model response {name} must be a finite number in [0, 1]')
    label = response['predicted_label']
    if isinstance(label, bool) or type(label) is not int or label not in (0, 1):
        raise ValueError('Model response predicted_label must be integer 0 or 1')
    if label != int(response['risk_score'] >= response['threshold']):
        raise ValueError('Model response predicted_label disagrees with risk_score/threshold')
    return response


def response_key(response):
    """Opaque Kafka key: transaction/model/feature version for backend upserts."""
    validate_response(response)
    return ':'.join(response[name] for name in KEY_FIELDS)
