classdef mRVM_train_output < handle
    
    properties
        model_used;
        dataset_name;
        N_total;
        N_prototypical;
        
        X_prototypical;
        
        X_prototypical_standardized;
        Xmean;
        Xstd;
        K_sparse;
        W_sparse; 
        
        active_sample_original_indices;
        
        sources;
        b;
        
        kernel_type;
        kernel_param;
        standardize_flag;
    end
    
    methods
        function obj = mRVM_train_output(model_used, dataset_name, N_total, X_prototypical,standardize_flag,X_prototypical_standardized, Xmean, Xstd, K_sparse, W_sparse,...
                active,kernel_type,kernel_param, sources, b)
            obj.model_used = model_used;
            obj.dataset_name = dataset_name;
            obj.N_total = N_total;
            obj.N_prototypical = length(active);
            obj.X_prototypical = X_prototypical;
            obj.X_prototypical_standardized = X_prototypical_standardized;
            obj.Xmean=Xmean;
            obj.Xstd=Xstd;
            obj.K_sparse = K_sparse;
            obj.W_sparse = W_sparse;
            obj.active_sample_original_indices = active;
            obj.kernel_type = kernel_type;
            obj.kernel_param = kernel_param;
            obj.standardize_flag = standardize_flag;
            
            if ~exist('sources','var')
                obj.sources = 1;
                obj.b = 1;
            else
                obj.sources = sources;
                obj.b = b;
            end
        end
    end
end