def scvi_pipeline(data,n_latents,n_layers=2,run_leiden=[1],
                  hvg_key="Batch",
                  batch_key="Batch",
                  n_top_genes=1000):
    
    """
    Perform a complete run of HVG + data reduction + scVI + leiden for a given 
    andata 
    ---------------- Parameter ------------------------
    data: an anndata where .X keep the raw count for ALL GENES
    hvg_key: our strategy to run highly variable genes, there are three options
    for this:
        - Batch: run hvg per Batch and then concatenate the list 
        - Disease: run the above strategy (Batch), for each Disease, and concatenate 
                    the list across Diseases
    batch_key: the Batch key to be found in data.obs
    n_top_genes: number of HVG genes to be used for downstream 
    ----------------- Output -------------------------------
    adata: an anndata with UMAP embedding,
           de_res result for the given leiden resoluation, 
           and .raw.X contain raw counts 
           .X contains normalized  counts 
    """

    assert np.max(data.X[0:1000,0:1000]).is_integer()
    assert hvg_key  in ["Disease","Batch","All"]
    assert batch_key  in data.obs
    if hvg_key=="Disease":
        assert "Disease" in data.obs
    assert isinstance(run_leiden, list)

    ### Assuming that data is not normalized:
    dt_tmp=data.copy()
    sc.pp.normalize_total(dt_tmp)
    sc.pp.log1p(dt_tmp)
    ### We have to make sure that division of zero doesnt exist when 
    ### a few batches have one cell:
    print("Step1: HVGs selection")
    if hvg_key=="Batch":    
        print("Run HVG per Batch")
        batch_ncell=dt_tmp.obs["Batch"].value_counts()
        to_keep=batch_ncell.index[batch_ncell>1]
        
        dt_tmp_var=sc.pp.highly_variable_genes(dt_tmp[dt_tmp.obs["Batch"].isin(to_keep)],
                                    batch_key="Batch",
                                    inplace=False,
                                    n_top_genes=n_top_genes)
        dt_tmp.var.robust=np.invert(dt_tmp.var_names.isin(bcr["Approved symbol"]))
        selected=np.logical_and(dt_tmp_var.highly_variable,dt_tmp.var.robust)
    elif hvg_key=="All":
        print("Find HVGs not using Batch information")

        sc.pp.highly_variable_genes(dt_tmp,n_top_genes=n_top_genes)
        dt_tmp.var.robust=np.invert(dt_tmp.var_names.isin(bcr["Approved symbol"]))

        selected=np.logical_and(dt_tmp.var.highly_variable,dt_tmp.var.robust)

    elif hvg_key=="Disease":
        print("Find HVGs per Disease and concatenate:")

        batch_ncell=[dt_tmp[dt_tmp.obs["Disease"]==i].obs["Batch"].value_counts()
                     for i in set(dt_tmp.obs["Disease"])]
        
        to_keep_per_Disease=[i.index[i>1] for i in batch_ncell]
        highly_vars=[ sc.pp.highly_variable_genes(dt_tmp[np.logical_and(dt_tmp.obs["Disease"]==i,dt_tmp.obs["Batch"].isin(j))] ,
                                      batch_key="Batch",inplace =False,
                                      n_top_genes) for i,j in zip(set(dt_tmp.obs["Disease"]),
                                                                                       to_keep_per_Disease)]
        highly_vars=np.unique(np.hstack([i.index[i["highly_variable"]] for i in highly_vars]))
        dt_tmp.var.robust=np.invert(dt_tmp.var_names.isin(bcr["Approved symbol"]))

        
        selected=np.logical_and(dt_tmp.var_names.isin(highly_vars),
                                dt_tmp.var.robust)

    dt_tmp=dt_tmp[:,selected].copy()
    dt_tmp.layers["counts"]=data[:,dt_tmp.var_names]
    print("Step 2: Set up scVI model")
    scvi.model.SCVI.setup_anndata(dt_tmp,  batch_key="Batch")
    vae = scvi.model.SCVI(dt_tmp, n_layers=n_layers, n_latent=n_latents, gene_likelihood="nb")
    vae.train()
    dt_tmp.obsm["X_scVI"] = vae.get_latent_representation()
    print("Step 3: Run dimension reduction:")
    sc.pp.neighbors(dt_tmp, use_rep="X_scVI",method="rapids")
    
    sc.tl.umap(dt_tmp,method="rapids")
    obs=dt_tmp.obs
    non_old_leiden_columns=obs.columns[np.invert(obs.columns.str.contains("leiden_res"))]
    obs=obs[non_old_leiden_columns]
    
    
    obsp=dt_tmp.obsp
    obsm=dt_tmp.obsm
    dt_tmp=data.copy()
    dt_tmp.raw=dt_tmp
    sc.pp.normalize_total(dt_tmp)
    sc.pp.log1p(dt_tmp)
    dt_tmp.obs=obs
    dt_tmp.obsm=obsm
    dt_tmp.obsp=obsp
    print("Step 4: Run leiden clustering:")
    for j in  run_leiden:

        dt_tmp.obs["leiden_res_"+str(j)+"_scVI"]=leiden(dt_tmp,j)
        pg.de_analysis(dt_tmp,"leiden_res_"+str(j)+"_scVI",de_key="de_res_"+str(j)+"_scVI")


    return(dt_tmp) 
def leiden(adata, resolution=1.0):
    """
    Performs Leiden Clustering using cuGraph

    Parameters
    ----------

    adata : annData object with 'neighbors' field.

    resolution : float, optional (default: 1)
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.

    """
    import cudf
    import cugraph
    # Adjacency graph
    adjacency = adata.obsp['connectivities']
    offsets = cudf.Series(adjacency.indptr)
    indices = cudf.Series(adjacency.indices)
    g = cugraph.Graph()
    if hasattr(g, 'add_adj_list'):
        g.add_adj_list(offsets, indices, None)
    else:
        g.from_cudf_adjlist(offsets, indices, None)

    # Cluster
    leiden_parts, _ = cugraph.leiden(g,resolution = resolution)

    # Format output
    clusters = leiden_parts.to_pandas().sort_values('vertex')[['partition']].to_numpy().ravel()
    clusters = pd.Categorical(clusters)

    return clusters