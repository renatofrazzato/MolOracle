import numpy as np

def GetFeaturesPaddingAdjustments(param_mol_sizes, param_max_size):
  lst_bias_features = list()
  for c in param_mol_sizes:
    mol_bias_vec = np.concatenate([np.ones([c,1]), np.zeros([param_max_size-c,1])], 0)
    lst_bias_features.append(mol_bias_vec)

  return np.array(lst_bias_features)


def GetDistPaddingAdjustments(param_batch_mol_size, param_max_atm):
  lst_batch_result = list()
  for i in range(len(param_batch_mol_size)):
    lst_batch_result_aux = list()
    for r in range(param_batch_mol_size[i]):
      vec_aux = np.ones((param_batch_mol_size[i],1))
      vec_aux[r,0] = 0.0
      vec_aux_zeros = np.zeros((param_max_atm-param_batch_mol_size[i],1))
      vec_aux = np.concatenate([vec_aux,vec_aux_zeros],0)
      lst_batch_result_aux.append(vec_aux)

    for r in range(param_max_atm-param_batch_mol_size[i]):
      vec_aux = np.zeros((param_max_atm,1))
      lst_batch_result_aux.append(vec_aux)

    lst_batch_result.append(lst_batch_result_aux)
  return np.array(lst_batch_result)


def GerarBatch(p_features, p_target, p_distancias, p_mol_sizes, n_batch, param_max_size):
  lst_batch_features = list()
  lst_batch_dist = list()
  lst_batch_target = list()
  aux_batch_mol_size = list()

  idx_selecao = np.random.choice([x for x in range(len(p_target))], n_batch, replace=False)

  for idx in idx_selecao:
    val_target = p_target[idx]
    val_dist = p_distancias[idx]
    val_mol_size = p_mol_sizes[idx]
    val_atm_features = p_features[idx]


    lst_batch_features.append(val_atm_features)
    lst_batch_target.append([val_target])
    lst_batch_dist.append(val_dist)
    aux_batch_mol_size.append(val_mol_size)


  lst_batch_features_padding = GetFeaturesPaddingAdjustments(aux_batch_mol_size, param_max_size)
  lst_batch_cfconv_padding = GetDistPaddingAdjustments(aux_batch_mol_size, param_max_size)

  return np.array(lst_batch_features), np.array(lst_batch_dist), np.array(lst_batch_target), lst_batch_features_padding, lst_batch_cfconv_padding