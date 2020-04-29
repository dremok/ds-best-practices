from sacred.observers import FileStorageObserver, MongoObserver

mongo_observer = MongoObserver(url='localhost:27017', db_name='db')

file_observer = FileStorageObserver('house_prices')
