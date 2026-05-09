import ee
ee.Initialize(project='famm-472015')

# List all tasks
tasks = ee.batch.Task.list()

print(f"Found {len(tasks)} tasks")

# Cancel all tasks
for t in tasks:
    if t.active():  # only running or ready tasks
        print(f"Cancelling task: {t.id} | {t.state} | {t.config['description']}")
        t.cancel()

