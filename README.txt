0. This is the source code of the paper "A New Index for Clustering Evaluation Based on Density Estimation."

1. In function index_plot_first_n_label_one_data, if the index's score is "smaller is better", then the "smaller_better" hyper-parameter should be set to True. Otherwise, if the index's score is "larger is better", then the "smaller_better" hyper-parameter should be set to False.

2. Readers can test their own index function, the API is:
   
    def index_function(X, label):
   
    some codes to compute the index value ...
    
    return the_index_value

then call the index_plot_first_n_label_one_data function. Note the "smaller_better" hyper-parameter.

3. License.
   
License of the source code : Apache License, Version 2.0
License of new data: Creative Commons Attribution 4.0 International

4. Citation:

@article{liu2022new,
  title={A new index for clustering evaluation based on density estimation},
  author={Liu, Gangli},
  journal={arXiv preprint arXiv:2207.01294},
  year={2022}
}

5. The "multiple_label_145.p" file is larger than 100MB, so it is stored on Git Large File Storage (LFS), readers may need to download it separately.

 




 
