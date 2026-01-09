import requests
# def factory_metric():
#         return {'key': None, 'label':[], 'value': 0}
# res = defaultdict(factoryheader)
class Metrics(object):
    """
    Peer Metrics 解释器
    """

    def __init__(self, url, filters:list):
        """
        param: filter: metrics参数过滤器
        """
        self._filter = filters
        self._url = url
    
    @property
    def schema(self):
        return {'key': None, 'label':{}, 'value': 0}
        

    def interprete(self) -> dict:
        """
        解析
        """
        metrics = []
        
        try:
            response = requests.request("GET", self._url, timeout=5)
            data = response.text
        except Exception as e:
            print(f"Error fetching metrics from {self._url}: {e}")
            return []

        lines = data.split('\n')
        for line in lines:
            if line.startswith('# ') or len(line) == 0:
                continue

            try:
                metric_item = self.schema.copy() # Use copy to avoid reference issues
                # factory_metric returns a new dict, but here self.schema is a property returning a new dict? 
                # Property implementation: return {'key': None, 'label':{}, 'value': 0} -> Yes, it returns a new dict.
                # But let's be safe.
                
                line_parts = line.split(' ')
                if len(line_parts) < 2:
                    continue
                    
                metric_item['value'] = float(line_parts[1])
                
                if line_parts[0].endswith('}'):
                    metric_item['key'] = line_parts[0].split('{')[0]
                    # Handle labels
                    label_part = line_parts[0].split('{')[1][:-1]
                    if label_part:
                        labels = label_part.split(',')
                        for la in labels:
                            if '=' in la:
                                key, val = la.split('=', 1)
                                metric_item['label'][key] = val.strip('"')
                else:
                    metric_item['key'] = line_parts[0]

                metrics.append(metric_item)
            except Exception as e:
                # print(f"Error parsing line '{line}': {e}")
                continue

        # filter
        if self._filter:
            result = []
            for item in metrics:
                if item['key'] in self._filter:
                    result.append(item)
            
            return result

        return metrics




    