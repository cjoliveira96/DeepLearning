# Imports
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import tensorflow as tf
import seaborn as sns


# Function to plot bar charts
def bar_chart(data, variable, x_label, fontsize=10):
    # Extract the values
    counts = data[variable].value_counts()
    x, y = counts.index, counts.values

    # Plot the bar chart
    plt.bar(x, y) 

    # Include the values in the plot
    for i, v in enumerate(y):
        plt.text(i, v, str(v), ha='center', va='bottom', fontsize=fontsize) 

    # Set x and y labels
    plt.xlabel(x_label)
    plt.ylabel('Count') 


# Function to remove hair from images
def dullrazor(img):

    # Gray scale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Black hat filter
    kernel = cv2.getStructuringElement(1, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # Gaussian filter
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)

    # Binary thresholding (MASK)
    ret, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)

    # Replace pixels of the mask
    dst = cv2.inpaint(img, mask, 6, cv2.INPAINT_TELEA)

    return dst


# Remove hair, resize, normalize (and adjust brightness/contrast) of images 
def inicial_preproc(data: list, alpha=1.0, beta=0):
    
    preprocessed_images = []

    for img in data:

        img_dullrazor = dullrazor(img)
        
        img_resized = cv2.resize(img_dullrazor, (71, 71))

        # img_adjusted = cv2.convertScaleAbs(img_resized, alpha=alpha, beta=beta)

        img_normalized = np.array(img_resized) / 255.0  # Normalize to [0, 1] range
        
        preprocessed_images.append(img_normalized)

    preprocessed_images = np.array(preprocessed_images)

    return preprocessed_images


# Function to plot the before and after preprocessing
def plot_befor_after(metadata_train_over, images_data_train, X_train_over):

    # Select a random row from the metadata dataframe for its image to be plotted
    row = metadata_train_over.sample(n=1)

    # Get the index of the image in the original metadata and in the current one
    after_idx = row.index[0]
    before_idx = row['original_train_index'].values[0]
    
    # Extract and normalize the original image
    normalized_img = np.array(images_data_train[before_idx]) / 255.0

    # Extract the corresponding image from the current dataset
    img = X_train_over[after_idx]

    # Create a subplot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Plot the original image
    axs[0].imshow(normalized_img)
    axs[0].set_title('Original Image')

    # Plot the current image
    axs[1].imshow(img)
    axs[1].set_title('New Image')

    # Display the plot
    plt.show()


# Function to print performance metrics
def metrics(y_train, pred_train , y_val, pred_val):
    print('___________________________________________________________________________________________________________')
    print('                                                     TRAIN                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(classification_report(y_train, pred_train))
    print(accuracy_score(y_train, pred_train))
    print(confusion_matrix(y_train, pred_train))


    print('___________________________________________________________________________________________________________')
    print('                                                VALIDATION                                                 ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(classification_report(y_val, pred_val))
    print(accuracy_score(y_val, pred_val))
    print(confusion_matrix(y_val, pred_val))


# Function to print performance metrics only for one set
def metrics_test(y_test, pred_test):
    print('___________________________________________________________________________________________________________')
    print('                                                    TEST                                                   ')
    print('-----------------------------------------------------------------------------------------------------------')
    print(classification_report(y_test, pred_test))
    print(accuracy_score(y_test, pred_test))


# Function to plot the results of the model
def plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):

    fig, ax = plt.subplots(figsize=(4, 3))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]

    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, 7])
    plt.ylim(ylim)

    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.close()


# Function to plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred):

    # Define the class names
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a DataFrame from the confusion matrix for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))

    # Create a heatmap from the DataFrame
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')

    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Function to plot the images and their predictions and true labels
def evaluate_model(X_test, y_test, model):
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    num_cols = 6  # Number of images per class to display

    num_total = 0
    num_correct = 0

    # Create a figure to display the images
    plt.figure(figsize=(15, len(class_names)*3))  # Adjust the size based on your needs

    # Iterate over each class
    for i, class_name in enumerate(class_names):
        # Get indices of samples belonging to the current class
        class_idxs = np.where(y_test == i)[0]

        # Randomly select 'num_cols' samples from the current class
        idxs = np.random.choice(class_idxs, size=num_cols, replace=False)
        data_batch = X_test[idxs]

        # Get predictions from model
        last_predictions = model.predict(data_batch)

        for j in range(num_cols):
            ax = plt.subplot(len(class_names), num_cols, i*num_cols + j + 1)
            plt.axis("off")
            plt.imshow(data_batch[j])

            pred_idx = tf.argmax(last_predictions[j]).numpy()
            truth_idx = y_test[idxs[j]]

            title = str(class_names[truth_idx]) + " : " + str(class_names[pred_idx])
            title_obj = plt.title(title, fontdict={'fontsize':13})

            if pred_idx == truth_idx:
                plt.setp(title_obj, color='g')
                num_correct += 1
            else:
                plt.setp(title_obj, color='r')

            num_total += 1

    plt.tight_layout()
    plt.show()

    # Calculate and print the overall accuracy
    overall_accuracy = num_correct / num_total
    print(f"Overall accuracy: {overall_accuracy:.2f}")

    return

