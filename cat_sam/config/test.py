import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Example usage
if __name__ == "__main__":
    file_path = './Reins_Attention6.yaml'
    yaml_data = read_yaml(file_path)
    print(yaml_data)
    print(yaml_data['model']['reins_config']['if_evp_feature'] == False)