# from typing import List
# # from scipy.sparse import csr_matrix, hstack
# # from app.adapters.qdrant import from_sparse_vector
# import numpy as np


# class SparseVector:
#     pass

#     @property
#     def shape(self):
#         return self.mat.shape[1]

#     def __init__(self, csr_mat: csr_matrix, dimentions: List[int] =None, dim_labels: List[str]=None):        
#         self.dimentions = dimentions
#         self.dim_labels = dim_labels
#         if type(csr_mat) != csr_matrix:
#             raise ValueError("Only CSR matrix is supported")
#         self.mat = csr_mat
#         self.agg_dimentions = [sum(dimentions[:i]) for i in range(1, len(dimentions)+1)]
#         self.dim_labels_lookup = {}
#         for i, lbl in enumerate(dim_labels):
#             self.dim_labels_lookup[lbl] = {
#                 "range": [self.agg_dimentions[i-1] if i > 0 else 0, self.agg_dimentions[i]],
#                 "size": dimentions[i]
#             }

#     def __getitem__(self, key):
#         if isinstance(key, list) and all(isinstance(k, str) for k in key):
#             idxs = [(0, range(self.dim_labels_lookup[k]["range"][0], self.dim_labels_lookup[k]["range"][1])) for k in key]
#             dimentions = [d for  l,d in zip(self.dim_labels, self.dimentions) if l in key]            
#             mat = hstack([self.mat[i] for i in idxs])
#             return SparseVector(mat, dimentions=dimentions, dim_labels=key)
#         if isinstance(key, slice):
#             mat = self.mat[0, key]
#         if isinstance(mat, csr_matrix):            
#             return SparseVector(mat, dimentions=self.dimentions, dim_labels=self.dim_labels)
#         return mat
    
#     def get_bins(self):
#         _, col = self.mat.nonzero()
#         x_ticks = [sum(self.dimentions[:i]) for i in range(1, len(self.dimentions)+1)]    
#         return [(c, self.dim_labels[np.digitize(c, x_ticks)], self.mat[0,c]) for c in col]

#     def values(self):
#         return self.get_bins()
#         # return [np.digitize(c, x_ticks) for c in col]

#     def show(self):
#         import seaborn as sns
#         import matplotlib.pyplot as plt
#         x_ticks = [sum(self.dimentions[:i]) for i in range(1, len(self.dimentions)+1)]
#         plt.figure(figsize=(30, 1))
#         # ax = sns.heatmap(self.mat.todense(), cmap='viridis', cbar=True, fmt=".1f", linewidths=.0028)
#         ax = sns.heatmap(self.mat.todense(), cmap='viridis', fmt=".1f", linewidths=.0028)

#         # Set custom tick labels
#         # ax.set_xticks(np.arange(0, cols, step=range_step) + range_step / 2)
#         ax.set_xticks(x_ticks)
#         ax.set_xticklabels(self.dim_labels)
#         plt.title('Sparse Vector Heatmap')
#         plt.xlabel('Column Ranges')
#         # plt.ylabel('Rows')
#         # plt.xticks(rotation=45)
#         plt.show()


#     def multiply(self, other):
#         if other.shape != self.shape:
#             raise ValueError("Shapes do not match")
#         elif isinstance(other, SparseVector):
#             other = other.mat
#         mat = self.mat.multiply(other)
#         return SparseVector(mat, dimentions=self.dimentions, dim_labels=self.dim_labels)


#     def to_data(self, sparse_vec):
#         _, col = self.mat.nonzero()    
#         return col.tolist(), sparse_vec.data.tolist()


#     @staticmethod
#     def from_data(data, dim, dimentions=None, dim_labels=None):
#         rows = np.zeros(len(data['indices']))  # Assuming all values are in a single row
#         cols = data['indices']
#         values = data['values']
#         return SparseVector(csr_matrix((values, (rows, cols)), shape=(1, dim)), dimentions=dimentions, dim_labels=dim_labels)


#     def show2(self):
#         from IPython.display import display, HTML
#         values = self.get_bins()
# #         html = f"""
# # <div style="display: flex;">
# #     {"<br>".join([f"<div style='width: 100px; text-align: center; padding: 10px; border: 1px solid #ccc; margin: 5px;'>{v[1]}<br>{v[2]}</div>" for v in values])}
# # </div>
# # """     
#         values = [f"<tr><td>{v[0]}</td><td>{v[1]}</td><td style='width: 200px;'>{round(v[2], 2)}</td></tr>" for v in values]
#         html = f"""
# <table >
#     <tr>
#         <th>Column</th>
#         <th>Label</th>
#         <th>Value</th>
#     </tr>
#     {" ".join(values)}
# </table>
# """     

#         display(HTML(html))

#     def get_pandas_df(self):
#         import pandas as pd
#         values = self.get_bins()
#         return pd.DataFrame(values, columns=["Column", "Label", "Value"])

#     def show3(self):
#         import pandas as pd
#         values = self.get_bins()

#         return pd.DataFrame(values, columns=["Column", "Label", "Value"])