import pandas as pd 
import flask
import pickle
import numpy as np
import portalocker


# Use pickle to load in the pre-trained model.
with open('model/SVM.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('model/Mapping.pkl', 'rb') as fp:
    mapping = pickle.load(fp)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        
         # Extract the input
        asin = flask.request.form['asin']
        initial_identifier = flask.request.form['initial_identifier']
        title = flask.request.form['title']
        brand = flask.request.form['brand']
        manufacturer = flask.request.form['manufacturer']
        publisher = flask.request.form['publisher']
        studio = flask.request.form['studio']
        label = flask.request.form['label']
        features = flask.request.form['features']
        all_categories = flask.request.form['all_categories']
        nodeid_tree = flask.request.form['nodeid_tree']
        subcategory = flask.request.form['subcategory']
        part_num = flask.request.form['part_num']
        mpn = flask.request.form['mpn']
        product_type_name = flask.request.form['product_type_name']
        category = flask.request.form['category']
        model = flask.request.form['model']
        sku = flask.request.form['sku']
        product_group = flask.request.form['product_group']
        parent_asin = flask.request.form['parent_asin']
        color = flask.request.form['color']
        binding = flask.request.form['binding']
        size = flask.request.form['size']
        package_dimensions_width = flask.request.form['package_dimensions_width']
        package_dimensions_length = flask.request.form['package_dimensions_length']
        
        # Make DataFrame for model
        input_variables = pd.DataFrame([[initial_identifier, asin, title,                         
                                        brand, manufacturer, publisher,                      
                                        studio, label, features,
                                        all_categories, nodeid_tree,                   
                                        subcategory, part_num, mpn,                                        
                                        product_type_name, category,     
                                        model, sku, product_group,                 
                                        parent_asin, color, binding, size, 
                                        package_dimensions_width, package_dimensions_length]],
                                        columns=['initial_identifier', 'asin',                
                                        'title', 'brand', 'manufacturer',                   
                                        'publisher','studio', 'label',                         
                                        'features', 'all_categories',                
                                        'nodeid_tree', 'subcategory','part_num',                      
                                        'mpn', 'product_type_name','category',                      
                                        'model', 'sku', 'product_group',                 
                                        'parent_asin', 'color','binding', 'size',
                                        'package_dimensions_width','package_dimensions_length'],
                                        dtype=object,
                                        index=['input'])  
        
        classifier_variables=['initial_identifier', 'asin',                
                              'title', 'brand', 'manufacturer',                   
                              'publisher','studio', 'label',                         
                              'features', 'all_categories',                
                              'nodeid_tree', 'subcategory','part_num',                                
                              'mpn', 'product_type_name','category',                      
                              'model', 'sku', 'product_group',                 
                             'parent_asin', 'color','binding', 'size']
        quantative_variables=['package_dimensions_width', 'package_dimensions_length' ]

        for j in classifier_variables:       
            input_variables[j].replace('', value="UNKNOWN", inplace=True)

        for k in quantative_variables:
            input_variables[k].replace('', value=0, inplace=True)
            
        
        

        # print(input_variables) 
        #encoding and assigning mapping 
        classifier_encoded_variables=[]
        for val in classifier_variables:
            classifier_encoded_variables.append(val+'_En')
            uval=input_variables[val].unique()
            for i in uval:
                if i in mapping[val]:
                    input_variables.loc[input_variables[val]==i,val+'_En'] = mapping[val][i]
                else:
                    input_variables.loc[input_variables[val]==i,val+'_En'] = 0.5
        
        general_features=classifier_encoded_variables+quantative_variables
        
        input_data=input_variables[general_features]
        # Get the model's prediction        
        prediction = svm_model.predict(input_data)[0]
      
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',original_input={},result=prediction,)
if __name__ == '__main__':
    app.run()