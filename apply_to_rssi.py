import helper_functions as hf

helper_variables = hf.main('rssi')
report_card_df = helper_variables[1]
columns_to_score = helper_variables[2]

normalized_dfs = []

for column in columns_to_score:
    normalized_df = hf.normalize_scores(report_card_df.copy(), column, bin_size=100)
    normalized_df.rename(columns={f'{column}_score_ln_normal_100': f'{column}_TALLY_values'}, inplace=True)
    normalized_dfs.append(normalized_df[['rcdts', column, f'{column}_TALLY_values']])

report_card_df = report_card_df.drop(['disciplinary_action_ratio', 'violent_incidents_ratio'], axis=1)

for norm_df in normalized_dfs:
    report_card_df = report_card_df.merge(norm_df, on='rcdts', how='left')

selected_columns = ['rcdts', 'school_name']

report_card_df.rename(columns={
    'disciplinary_action_ratio': 'disciplinary_action_ratio_RAW_values',
    'violent_incidents_ratio': 'violent_incidents_ratio_RAW_values'
}, inplace=True)

report_card_df = report_card_df[selected_columns + ['disciplinary_action_ratio_RAW_values', 
                                                     'violent_incidents_ratio_RAW_values', 
                                                     'disciplinary_action_ratio_TALLY_values',
                                                     'violent_incidents_ratio_TALLY_values']]

report_card_df.to_csv('report_card_log_normalized.csv', index=False)
