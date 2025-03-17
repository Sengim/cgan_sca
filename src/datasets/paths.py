def get_dataset_filepath(dataset_root_folder, dataset_name, npoi, leakage_model):
    if dataset_name == "simulate":
        return "sim"
    if leakage_model == "ID":
        dataset_dict = {
            "ASCAD": {
                # 100: f"{dataset_root_folder}/ASCADf/ASCAD_rpoi/ASCAD_100poi.h5",
                700: f"{dataset_root_folder}/ASCAD.h5",
                20000: f"{dataset_root_folder}/ASCADf/ASCAD_nopoi/ASCAD_nopoi_window_10.h5",
                10000: f"{dataset_root_folder}/ASCADf/ASCAD_nopoi/ASCAD_nopoi_window_20.h5",
                5000: f"{dataset_root_folder}/ASCADf/ASCAD_nopoi/ASCAD_nopoi_window_40.h5",
                2500: f"{dataset_root_folder}/ASCADf/ASCAD_nopoi/ASCAD_nopoi_window_80.h5"
            },
            "ascad-variable": {
                # 100: f"{dataset_root_folder}/ASCADr/ascad-variable_rpoi/ascad-variable_100poi.h5",
                1400: f"{dataset_root_folder}/ascad-variable.h5",
                50000: f"{dataset_root_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_10.h5",
                25000: f"{dataset_root_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_20.h5",
                12500: f"{dataset_root_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_40.h5",
                6250: f"{dataset_root_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_80.h5",
                250000: f"{dataset_root_folder}/ASCADr/atmega8515-raw-traces.h5",
            },
            "dpa_v42": {
                # 100: f"{dataset_root_folder}/DPAV42/DPAV42_rpoi/dpa_v42_100poi.h5",
                30000: f"{dataset_root_folder}/DPAV42/DPAV42_nopoi/dpa_v42_nopoi_window_10.h5",
                15000: f"{dataset_root_folder}/DPAV42/DPAV42_nopoi/dpa_v42_nopoi_window_20.h5",
                7500: f"{dataset_root_folder}/DPAV42/DPAV42_nopoi/dpa_v42_nopoi_window_40.h5",
                3750: f"{dataset_root_folder}/DPAV42/DPAV42_nopoi/dpa_v42_nopoi_window_80.h5"

            },
            "ascadv2": {
                2000: f"{dataset_root_folder}/ascadv2-extracted.h5",
            },
            "eshard": {
                100: f"{dataset_root_folder}/ESHARD/ESHARD_rpoi/eshard_100poi.h5",
                1400: f"{dataset_root_folder}/eshard.h5",
            },
            "aes_sim_reference": {
                100: f"{dataset_root_folder}/aes_sim_mask_reference.h5",
            },
            "aes_sim_target": {
                100: f"{dataset_root_folder}/aes_sim_mask_target.h5",
            },
            "ches_ctf": {
                2200: f"{dataset_root_folder}/ches_ctf.h5",
            },
            "aes_hd_mm": {
                3125: f"{dataset_root_folder}/aes_hd_mm.h5",
            },

            "spook_sw3": {
                6250: f"D:/Datasets/sw3/spook_sw3_nopoi_window_20.h5.",
        },
        }
    else:
        dataset_dict = {
            "ASCAD": {
                # 100: f"{dataset_root_folder}/ASCADf/ASCAD_rpoi/ASCAD_100poi_hw.h5",
                700: f"{dataset_root_folder}/ASCAD.h5",
                20000: f"{dataset_root_folder}/ASCADf/ASCAD_nopoi/ASCAD_nopoi_window_10.h5",
                10000: f"{dataset_root_folder}/ASCADf/ASCAD_nopoi/ASCAD_nopoi_window_20.h5",
                5000: f"{dataset_root_folder}/ASCADf/ASCAD_nopoi/ASCAD_nopoi_window_40.h5",
                2500: f"{dataset_root_folder}/ASCADf/ASCAD_nopoi/ASCAD_nopoi_window_80.h5",
            },
            "ascad-variable": {
                # 100: f"{dataset_root_folder}/ASCADr/ascad-variable_rpoi/ascad-variable_100poi_hw.h5",
                1400: f"{dataset_root_folder}/ascad-variable.h5",
                50000: f"{dataset_root_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_10.h5",
                25000: f"{dataset_root_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_20.h5",
                12500: f"{dataset_root_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_40.h5",
                6250: f"{dataset_root_folder}/ASCADr/ascad-variable_nopoi/ascad-variable_nopoi_window_80.h5",
                250000: f"{dataset_root_folder}/ASCADr/atmega8515-raw-traces.h5",
            },
            "dpa_v42": {
                # 100: f"{dataset_root_folder}/DPAV42/DPAV42_rpoi/dpa_v42_100poi_hw.h5",
                30000: f"{dataset_root_folder}/DPAV42/DPAV42_nopoi/dpa_v42_nopoi_window_10.h5",
                15000: f"{dataset_root_folder}/DPAV42/DPAV42_nopoi/dpa_v42_nopoi_window_20.h5",
                7500: f"{dataset_root_folder}/DPAV42/DPAV42_nopoi/dpa_v42_nopoi_window_40.h5",
                3750: f"{dataset_root_folder}/DPAV42/DPAV42_nopoi/dpa_v42_nopoi_window_80.h5",
            },
            "ascadv2": {
                15000: f"{dataset_root_folder}/ascadv2-extracted.h5",
            },
            "eshard": {
                100: f"{dataset_root_folder}/ESHARD/ESHARD_rpoi/eshard_100poi_hw.h5",
                1400: f"{dataset_root_folder}/eshard.h5",
            },
            "aes_sim_reference": {
                100: f"{dataset_root_folder}/aes_sim_mask_reference.h5",
            },
            "aes_sim_target": {
                100: f"{dataset_root_folder}/aes_sim_mask_target.h5",
            },
            "ches_ctf": {
                2200: f"{dataset_root_folder}/ches_ctf.h5",
            },
            "aes_hd_mm": {
                3125: f"{dataset_root_folder}/aes_hd_mm.h5",
            }, 
            "spook_sw3": {
            6250: f"{dataset_root_folder}/sw3/spook_sw3_nopoi_window_20.h5."}
        }
    return dataset_dict[dataset_name][npoi]
