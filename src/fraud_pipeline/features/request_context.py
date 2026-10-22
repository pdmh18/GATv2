"""Optional request-time fields from reviewer; no IDs or terminal data as numeric inputs."""
OPTIONAL_FIELDS=('authentication_method','authentication_verified','device_id','platform')


def context_names(options):
    if not options.get('include_request_context',False):return []
    return (['authentication_verified','has_authentication_method','has_authentication_verified',
             'has_platform','has_device','has_device_history','new_device']+
            ['authentication_method_'+v for v in options['authentication_methods']]+
            ['platform_'+v for v in options['platforms']])


def validate_context_options(options):
    enabled=options.get('include_request_context',False)
    if type(enabled) is not bool:raise ValueError('include_request_context must be boolean')
    if enabled:
        for key in ('authentication_methods','platforms'):
            values=options[key]
            if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v or v!=v.lower() for v in values) or len(set(values))!=len(values) or 'unknown' not in values:
                raise ValueError(f'{key} must contain distinct lowercase categories including unknown')


def validate_context(payload):
    for key in OPTIONAL_FIELDS:
        value=payload.get(key)
        if value is None:continue
        if key=='authentication_verified':
            if type(value) is not bool:raise ValueError('authentication_verified must be bool or null')
        elif not isinstance(value,str) or not value.strip():
            raise ValueError(f'{key} must be a nonempty string or null')


def context_values(payload,prior_devices,options):
    validate_context(payload)
    device=payload.get('device_id');method=payload.get('authentication_method');platform=payload.get('platform')
    verified=payload.get('authentication_verified')
    method=method.lower() if method else 'unknown';platform=platform.lower() if platform else 'unknown'
    method=method if method in options['authentication_methods'] else 'unknown'
    platform=platform if platform in options['platforms'] else 'unknown'
    return ([float(verified) if verified is not None else 0.,float(payload.get('authentication_method') is not None),
             float(verified is not None),float(payload.get('platform') is not None),float(device is not None),
             float(bool(prior_devices)),float(device is not None and device not in prior_devices)]+
            [float(method==v) for v in options['authentication_methods']]+
            [float(platform==v) for v in options['platforms']])
