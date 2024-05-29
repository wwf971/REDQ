from tensorboard.backend.event_processing import event_accumulator

# Path to your TensorBoard log file
log_file_path = 'events.out.tfevents.1716735260'
    # 1716735260 is unix time stamp, seconds from 1970/01/01 00:00

# Initialize the EventAccumulator
ea = event_accumulator.EventAccumulator(log_file_path,
    size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
        event_accumulator.IMAGES: 4,
        event_accumulator.AUDIO: 4,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    })

# Load the events from the file
ea.Reload()

# Print the available tags (variables) in the log file
tags = ea.Tags()
print("Available tags:")
for tag_type in tags:
    print(f"{tag_type}: {tags[tag_type]}")