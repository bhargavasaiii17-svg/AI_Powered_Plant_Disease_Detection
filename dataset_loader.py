import tensorflow as tf

dataset_path = "dataset/PlantVillage/PlantVillage"

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

print("\nClasses:")
print(train_ds.class_names)

print(f"\nTotal Classes: {len(train_ds.class_names)}")