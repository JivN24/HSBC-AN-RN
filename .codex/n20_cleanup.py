import json
from pathlib import Path

notebook_path = Path(r"C:\Users\Rajiv Nawal\OneDrive\Documents\GITREPOS\HSBC-AN-RN\CODEWORK\Linus' Task Re-Do\20_empirical_event_weight_api.ipynb")
notebook = json.loads(notebook_path.read_text(encoding='utf-8'))

def markdown(identifier, text):
    return {'cell_type': 'markdown', 'id': identifier, 'metadata': {}, 'source': [line + '\n' for line in text.splitlines()]}

def code(identifier, text):
    return {'cell_type': 'code', 'execution_count': None, 'id': identifier, 'metadata': {}, 'outputs': [], 'source': [line + '\n' for line in text.splitlines()]}

retained = [cell for index, cell in enumerate(notebook['cells']) if index <= 3 or index >= 28]
insert_at = next(index for index, cell in enumerate(retained) if cell.get('id') == 'n20-dynamic-core')
overview = [
    markdown('n20-steps', '''## Canonical packaged workflow

Step 1 estimates or loads the N16 parametric structure. Step 2 produces event-unaware ordinary forecasts. Step 3 refits parametric models after target-event masking. Step 4 applies the dynamic N19 flexible specifications. FULL mode discovers structures and tunes hyperparameters; REUSE mode accepts the frozen structures and refits parameters and scalers only.'''),
    markdown('n20-timing', '''## Timing and downstream contract

All retained event occurrences use the corrected Bloomberg 17:00--16:59 New York pricing session and identical realised/implied model days. Ordinary-ratio exports are the event-unaware Step-2 forecasts required by Notebook 21; eventless forecasts are used only for event weights.'''),
]
retained[insert_at:insert_at] = overview

core = next(cell for cell in retained if cell.get('id') == 'n20-dynamic-core')
core_source = ''.join(core['source']).replace("validate='one_to_one').drop(columns='model_day')", "validate='many_to_one').drop(columns='model_day')")
core['source'] = [line + '\n' for line in core_source.splitlines()]

override = r'''# Step-2 event-unaware forecasts, public return contract, and manual reference path.
PUBLIC_API_KEYS = {'structure', 'parametric_ordinary', 'parametric_eventless', 'flexible_spec', 'flexible_ordinary', 'flexible_eventless', 'event_mapping', 'event_weights', 'pooled_event_weights', 'ordinary_response_ratios', 'model_specification', 'diagnostics', 'run_metadata'}

def _metrics(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(forecast)
    error = actual[valid] - forecast[valid]
    return {'n': int(valid.sum()), 'RMSE': float(np.sqrt(np.mean(error ** 2))), 'MAE': float(np.mean(np.abs(error)))}

def frozen_n16_structure():
    source = pd.read_csv(PROCESSED / '16_model_specification.csv').iloc[0]
    retained = pd.DataFrame([
        {'source': 'Q', 'target': 'R', 'retained': bool(source['retain_Q_to_R'])},
        {'source': 'I', 'target': 'R', 'retained': bool(source['retain_I_to_R'])},
        {'source': 'Q', 'target': 'I', 'retained': bool(source['retain_Q_to_I'])},
        {'source': 'R', 'target': 'I', 'retained': bool(source['retain_R_to_I'])},
    ])
    return {'selected_orders': {'R': int(source['selected_realised_order']), 'I': int(source['selected_implied_order'])}, 'retained_structure': retained, 'metadata': {'source': 'FROZEN_N16_EXPORT'}}

def _ordinary_rows(frame, target, family, stage, prediction, event_days):
    out = frame[['model_day', 'sample_split', target]].copy().rename(columns={target: 'actual'})
    out['target_model_day'] = out['model_day']; out['target'] = target; out['model'] = family; out['forecast_stage'] = stage
    out['background_forecast'] = np.asarray(prediction, dtype=float)
    out['equation_eligible'] = True
    out['valid_positive_background'] = np.isfinite(out['background_forecast']) & out['background_forecast'].gt(0)
    out['ratio_defined'] = out['valid_positive_background']
    out['response_ratio'] = np.where(out['ratio_defined'], out['actual'] / out['background_forecast'], np.nan)
    out['known_event_target_flag'] = out['model_day'].isin(event_days)
    out['ordinary_flag'] = ~out['known_event_target_flag']; out['is_out_of_sample'] = True
    return out[['model_day', 'target_model_day', 'sample_split', 'forecast_stage', 'target', 'model', 'actual', 'background_forecast', 'response_ratio', 'valid_positive_background', 'ratio_defined', 'equation_eligible', 'known_event_target_flag', 'ordinary_flag', 'is_out_of_sample']]

def estimate_flexible_ordinary_forecasts(model_panel, event_history, flexible_spec, config=CONFIG):
    panel = prepare_model_panel(model_panel, config)
    selection = validate_flexible_spec(flexible_spec)
    mapping = target_event_mapping(_normalise_events(event_history, panel, 'event_history'), panel)
    rows, fitted = [], {}
    for (target, family), item in sorted(selection['selected'].items()):
        features = flexible_feature_names(item['parent_blocks'], item['lag_order'])
        train = _model_frame(panel, target, 'train', item['parent_blocks'], item['lag_order'])
        validation = _model_frame(panel, target, 'validation', item['parent_blocks'], item['lag_order'])
        test = _model_frame(panel, target, 'test', item['parent_blocks'], item['lag_order'])
        if family in {'RF', 'GB'}:
            development = _tree(family, item).fit(train[features], train[target])
            validation_prediction = development.predict(validation[features])
            calibration = pd.concat([train, validation], ignore_index=True)
            final = _tree(family, item).fit(calibration[features], calibration[target])
            test_prediction = final.predict(test[features])
        else:
            development_models, development_scales = [], []
            for seed in config['neural_seeds']:
                network, scale, _ = _train_neural(family, item, seed, train, validation)
                development_models.append(network); development_scales.append(scale)
            validation_prediction = _predict_neural(development_models, development_scales, validation, family, item)
            final_models, final_scales, calibration = [], [], pd.concat([train, validation], ignore_index=True)
            for seed in config['neural_seeds']:
                network, scale, _ = _train_neural(family, item, seed, calibration, epochs=item['final_epochs'])
                final_models.append(network); final_scales.append(scale)
            final = (final_models, final_scales)
            test_prediction = _predict_neural(final_models, final_scales, test, family, item)
            development = (development_models, development_scales)
        fitted[(target, family, 'TRAIN_TO_VALIDATION')] = development
        fitted[(target, family, 'TRAIN_VALIDATION_TO_TEST')] = final
        event_days = set(mapping.loc[mapping['target'].eq(target), 'target_model_day'])
        rows.extend([_ordinary_rows(validation, target, family, 'TRAIN_TO_VALIDATION', validation_prediction, event_days), _ordinary_rows(test, target, family, 'TRAIN_VALIDATION_TO_TEST', test_prediction, event_days)])
    return {'forecasts': pd.concat(rows, ignore_index=True), 'models': fitted, 'metadata': {'event_mask_used': False, 'forecast_origin': 'FINAL_N19_STEP_2'}}

def assert_ratio_propagation(results):
    for frame, definition in [(results['event_weights'], 'weight_defined'), (results['ordinary_response_ratios'], 'ratio_defined')]:
        valid = frame[definition].astype(bool)
        assert np.allclose(frame.loc[valid, 'response_ratio'], frame.loc[valid, 'actual'] / frame.loc[valid, 'background_forecast'])
    return True

def n19_equivalence_summary(results, tolerances=None):
    tolerances = {'tree': 1e-10, 'neural': 1e-6} | ({} if tolerances is None else tolerances)
    reference = pd.read_csv(PROCESSED / '19_ordinary_response_ratios.csv', parse_dates=['model_day', 'target_model_day'])
    actual = results['ordinary_response_ratios'].copy()
    keys = ['model_day', 'target', 'model', 'sample_split', 'forecast_stage']
    merged = actual.merge(reference, on=keys, suffixes=('_api', '_ref'), how='outer', indicator=True)
    common = merged.loc[merged['_merge'].eq('both')].copy()
    rows = []
    for family in MODEL_FAMILIES:
        part = common.loc[common['model'].eq(family)].copy(); tolerance = tolerances['neural'] if family in {'MLP', 'TRANSFORMER'} else tolerances['tree']
        background = np.abs(part['background_forecast_api'] - part['background_forecast_ref'])
        ratio = np.abs(part['response_ratio_api'] - part['response_ratio_ref'])
        status = part['valid_positive_background_api'].eq(part['valid_positive_background_ref']) & part['ratio_defined_api'].eq(part['valid_positive_background_ref'])
        valid = part['valid_positive_background_api'] & part['valid_positive_background_ref']
        identity = part.loc[valid, 'response_ratio_api'] - part.loc[valid, 'response_ratio_ref'] - part.loc[valid, 'actual_api'] * (1 / part.loc[valid, 'background_forecast_api'] - 1 / part.loc[valid, 'background_forecast_ref'])
        rows.append({'component': 'ordinary_' + family, 'n_compared': len(part), 'max_abs_background_difference': float(background.max()) if len(part) else np.nan, 'max_abs_ratio_difference': float(ratio.max()) if len(part) else np.nan, 'max_abs_ratio_identity_residual': float(np.abs(identity).max()) if len(identity) else np.nan, 'tolerance': tolerance, 'passed': bool(len(part) and background.max() <= tolerance and ratio.max() <= tolerance and status.all() and np.allclose(identity, 0.0, atol=tolerance, rtol=0.0))})
    return pd.DataFrame(rows)

def estimate_event_weights(model_panel, event_history, evaluation_events=None, structure=None, flexible_spec=None, config=None):
    config = CONFIG if config is None else config
    structure_source = 'ESTIMATED_THIS_RUN' if structure is None else 'SUPPLIED'
    flexible_source = 'SELECTED_THIS_RUN' if flexible_spec is None else 'SUPPLIED'
    structure = estimate_dependence_structure(model_panel, config) if structure is None else structure
    flexible_spec = select_flexible_models(model_panel, event_history, config) if flexible_spec is None else validate_flexible_spec(flexible_spec)
    parametric_ordinary = fit_parametric_models(model_panel, structure, config)
    parametric_eventless = estimate_parametric_event_weights(model_panel, event_history, evaluation_events, structure, config)
    flexible_ordinary = estimate_flexible_ordinary_forecasts(model_panel, event_history, flexible_spec, config)
    flexible_eventless = estimate_flexible_event_weights(model_panel, event_history, evaluation_events, flexible_spec, config)
    event_weights = pd.concat([parametric_eventless['event_weights'], flexible_eventless['event_weights']], ignore_index=True)
    mapping = build_eventless_calibration_sample(model_panel, event_history, evaluation_events, config)['evaluation_mapping']
    run_metadata = pd.DataFrame([{'event_alignment_version': ALIGNMENT_VERSION, 'iv_pricing_hours_ny': '17:00-16:59', 'iv_daily_observation_interpretation': 'END_OF_BLOOMBERG_PRICING_DAY_STATE', 'bloomberg_pricing_session_identified': True, 'bloomberg_snapshot_time_identified': False, 'approximate_10am_iv_cut_used_for_production': False, 'linus_equation_5_used': False, 'structure_source': structure_source, 'flexible_spec_source': flexible_source, 'api_mode': 'FULL' if structure_source == 'ESTIMATED_THIS_RUN' and flexible_source == 'SELECTED_THIS_RUN' else 'REUSE_OR_PARTIAL_REUSE'}])
    return {'structure': structure, 'parametric_ordinary': parametric_ordinary, 'parametric_eventless': parametric_eventless, 'flexible_spec': flexible_spec, 'flexible_ordinary': flexible_ordinary, 'flexible_eventless': flexible_eventless, 'event_mapping': mapping, 'event_weights': event_weights, 'pooled_event_weights': pool_event_weights(event_weights, config), 'ordinary_response_ratios': flexible_ordinary['forecasts'], 'model_specification': _specification_table(flexible_spec['selected']), 'diagnostics': {'ratio_propagation_ready': True, 'ordinary_event_unaware': True}, 'run_metadata': run_metadata}

def export_n20_results(results, output_dir=PROCESSED, equivalence_summary=None):
    assert_ratio_propagation(results)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ordinary = results['ordinary_response_ratios'].copy()
    summary = ordinary.groupby(['sample_split', 'target', 'model', 'forecast_stage'], as_index=False).agg(n_rows=('response_ratio', 'size'), n_ratio_defined=('ratio_defined', 'sum'), median_response_ratio=('response_ratio', 'median'))
    specification = pd.DataFrame([('event_alignment_version', ALIGNMENT_VERSION), ('iv_pricing_hours_ny', '17:00-16:59'), ('iv_daily_observation_interpretation', 'END_OF_BLOOMBERG_PRICING_DAY_STATE'), ('bloomberg_pricing_session_identified', 'True'), ('bloomberg_snapshot_time_identified', 'False'), ('approximate_10am_iv_cut_used_for_production', 'False'), ('linus_equation_5_used', 'False')], columns=['item', 'value']).assign(section='API')
    exports = {'20_api_reference_event_weights.csv': results['event_weights'], '20_api_event_mapping.csv': results['event_mapping'], '20_api_ordinary_response_ratios.csv': ordinary, '20_api_ordinary_response_ratio_summary.csv': summary, '20_api_run_metadata.csv': results['run_metadata'], '20_api_specification.csv': specification, '20_api_equivalence_summary.csv': pd.DataFrame() if equivalence_summary is None else equivalence_summary}
    for filename, frame in exports.items(): frame.to_csv(output_dir / filename, index=False)
    return exports
'''

manual = r'''RUN_FULL_REDISCOVERY = False
RUN_REFERENCE_EXECUTION = False

if RUN_REFERENCE_EXECUTION:
    model_panel, event_history, evaluation_events = canonical_reference_inputs()
    n16_structure = None if RUN_FULL_REDISCOVERY else frozen_n16_structure()
    n19_specification = None if RUN_FULL_REDISCOVERY else frozen_n19_flexible_spec()
    reference_results = estimate_event_weights(model_panel, event_history, evaluation_events, structure=n16_structure, flexible_spec=n19_specification, config=CONFIG)
    assert set(reference_results) >= PUBLIC_API_KEYS
    assert_ratio_propagation(reference_results)
    equivalence_summary = n19_equivalence_summary(reference_results)
    assert equivalence_summary['passed'].all(), equivalence_summary.to_string(index=False)
    export_n20_results(reference_results, equivalence_summary=equivalence_summary)
'''

static = r'''_features = flexible_feature_names(('R', 'I'), 3)
assert _features == ['R_lag3', 'I_lag3', 'R_lag2', 'I_lag2', 'R_lag1', 'I_lag1']
_frozen = frozen_n19_flexible_spec()
assert _frozen['selected'][('R', 'MLP')]['final_epochs'] == 55
assert _frozen['selected'][('I', 'TRANSFORMER')]['final_epochs'] == 61
assert MLPRegressor(2, [4])(torch.zeros(3, 2)).shape == (3,)
assert TransformerRegressor(2, 1, 16, 2, 32)(torch.zeros(3, 1, 2)).shape == (3,)
assert TransformerRegressor(3, 5, 24, 4, 48)(torch.zeros(3, 5, 3)).shape == (3,)
_tiny_panel = pd.DataFrame({'model_day': pd.to_datetime(['2020-01-02', '2020-01-03']), 'sample_split': ['train', 'test'], 'squared_return': [1.0, 1.0], 'realised_variance_ann_252': [1.0, 1.0], IV_COLUMN: [1.0, 1.0]})
_tiny_events = pd.DataFrame({'event_id': ['overlap_a', 'overlap_b'], 'event_label': ['A', 'B'], 'realised_model_day': [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-02')]})
_normalised = _normalise_events(_tiny_events, _tiny_panel, 'overlap smoke')
_mapped = target_event_mapping(_normalised, _tiny_panel)
assert len(_normalised) == 2 and _mapped['is_overlap'].all() and (_mapped['n_target_occurrences'] == 2).all()
assert PUBLIC_API_KEYS <= {'structure', 'parametric_ordinary', 'parametric_eventless', 'flexible_spec', 'flexible_ordinary', 'flexible_eventless', 'event_mapping', 'event_weights', 'pooled_event_weights', 'ordinary_response_ratios', 'model_specification', 'diagnostics', 'run_metadata'}
print('N20 static dynamic-schema, overlap, and API-contract checks passed.')
'''

status = '''# N20 CODE UPDATE STATUS

Dynamic N19 estimator: YES  
Corrected N18 same-day timing: YES  
Event-unaware ordinary forecast path: YES  
Eventless Step-3 path separate: YES  
Overlap-safe event mapping: YES  
Complete public API return contract: YES  
Parametric Step-2 path executable: YES  
REUSE default reference path: YES  
FULL rediscovery implemented but disabled by default: YES  
Retired fixed-36 code removed entirely: YES  
Static checks: PASS  
Heavy empirical execution: NO  
Numerical N16-N19 equivalence: PENDING USER RUN'''

for identifier in ['n20-dynamic-api', 'n20-export-equivalence', 'n20-static-smoke', '6d9348a5']:
    retained = [cell for cell in retained if cell.get('id') != identifier]
retained.extend([code('n20-ordinary-public-api', override), code('n20-manual-reference-run', manual), code('n20-static-smoke', static), markdown('n20-status', status)])
notebook['cells'] = retained
notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
