import PySimpleGUI as sg
from gui.gui_logic import *
import io
import global_vars as gv
from PIL import Image
import numpy as np
import cv2
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

gv.patch_size = (32,128,128,1)
gv.unet_model_path = os.path.join(gv.CWD, 'unet_model_22_05_22_ne_128')
gv.mg_model_path = os.path.join(gv.CWD, 'mg_model_ne_10_06_22_5_0_new')
gv.organelle = "Nuclear-envelope" #"Tight-junctions" #Actin-filaments" #"Golgi" #"Microtubules" #"Endoplasmic-reticulum" 
#"Plasma-membrane" #"Nuclear-envelope" #"Mitochondria" #"Nucleolus-(Granular-Component)"

upper_layout = [
    [
        sg.Text("UNET Model:"),
        sg.Input(
            key='-UNET MODEL-',
            default_text=gv.unet_model_path,
            size=(100, 1)),
        sg.FileBrowse(target='-UNET MODEL-')
    ],
    [
        sg.Text("Mask Interpreter Model:"),
        sg.Input(
            key='-MI MODEL-',
            default_text=gv.mg_model_path,
            size=(100, 1)),
        sg.FileBrowse(target='-MI MODEL-')
    ],
    [
        sg.Text("Dataset File:"),
        sg.Input(
            key='-DATASET-',
            default_text=
            os.path.join( gv.CWD, "single_cell_training_from_segmentation/{}/image_list_train.csv".format(gv.organelle)),
            size=(100, 1)),
        sg.FileBrowse(target='-DATASET-'),
        sg.Button('Load')
    ]
]

middle_layout = [[
    sg.Text("Image Index:"),
    sg.Input(key='-IMAGE INDEX-', default_text='0'),
    sg.Text("Slice:"),
    sg.Slider(key='-SLICE-',
              enable_events=True,
              default_value=0,
              range=(0, 0),
              orientation='horizontal'),
    sg.Button('Show', disabled=True),
],[sg.Button('Plot pearson score by slice'),sg.Button('Plot weighted pearson score by slice')],
                 [
                     sg.Text("ROI mode:"),
                     sg.Radio("Rect",
                              default=True,
                              group_id="ROI",
                              key="-RADIO RECT-",
                              enable_events=True),
                     sg.Radio("Pixel",
                              group_id="ROI",
                              key="-RADIO PIXEL-",
                              enable_events=True),
                     sg.Radio("Full",
                              group_id="ROI",
                              key="-RADIO FULL-",
                              enable_events=True),
                     sg.Text("ROI:"),
                     sg.Text(key='-ROI TEXT-', text='')
                 ],
                 [sg.Text("Interperter:"),
                     sg.Radio("Gradcam",
                              default=True,
                              group_id="INTER",
                              key="-RADIO GC-",
                              enable_events=True),
                     sg.Radio("Saliency maps",
                              group_id="INTER",
                              key="-RADIO SM-",
                              enable_events=True),
                     sg.Radio("Guided backpropagtion",
                              group_id="INTER",
                              key="-RADIO GBP-",
                              enable_events=True),
                     sg.Radio("Mask Interpreter",
                              group_id="INTER",
                              key="-RADIO MI-",
                              enable_events=True)                     ],
                 [
                     sg.Text("Layers:"),
                     sg.Combo(key='-LAYERS-',
                              values=[],
                              auto_size_text=True,
                              enable_events=True,
                              size=(30, 10)),
                     sg.Checkbox('X grad cam',
                                 key='-X gradcam-',
                                 default=False,
                                 disabled=False,
                                 enable_events=True),
                     sg.Button('Calculate', disabled=True)
                 ],
                 [sg.Text("Mask TH:"),sg.Input(key='-NP TH-', default_text=0.5,size=(5, 1)),sg.Button('Noise prediction', disabled=True)],
                 [sg.Button('Evaluate interperters', disabled=True)]
                 ]


def create_images_layout():
    images_layout = [
        [
            sg.Frame(
                "Original View",
                layout=[[sg.Text("Status:"),
                         sg.Text(key='-STATUS-')],
                        [sg.Image(key='-MAIN IMAGE-', enable_events=True)]]),
            sg.Frame("History",
                     layout=[[
                         sg.Combo(key='-HISTORY-',
                                  values=[],
                                  auto_size_text=True,
                                  enable_events=True,
                                  size=(100, 10)),
                         sg.Button('Show History',
                                   disabled=True,
                                   enable_events=True)
                     ], [sg.Image(key='-HISTORY IMAGE-', enable_events=True)]])
        ],
        [
            sg.Frame("Views",
                     layout=[[
                         sg.Checkbox('DNA',
                                     key='-NUC CHECK-',
                                     default=False,
                                     disabled=True,
                                     enable_events=True),
                         sg.Text("TH:"),
                         sg.Input(key='-DNA TH-',
                                  default_text=0.5,
                                  size=(5, 1)),
                         sg.Checkbox('Membrane',
                                     key='-MEM CHECK-',
                                     default=False,
                                     disabled=True,
                                     enable_events=True),
                         sg.Text("TH:"),
                         sg.Input(key='-MEM TH-',
                                  default_text=0.5,
                                  size=(5, 1)),
                         sg.Text("Gamma:"),
                         sg.Input(key='-GAM TH-',
                                  default_text=0.5,
                                  size=(5, 1)),
                         sg.Text("Pearson score of slice:"),
                         sg.Input(key='-PEARSON-',
                                  default_text=0.0,
                                  size=(15, 1),
                                  readonly=True),
                         sg.Text("Weighted Pearson score:"),
                         sg.Input(key='-WEIGHT PEARSON-',
                                  default_text=0.0,
                                  size=(15, 1),
                                  readonly=True),
                         sg.Text("Pearson score of slice - noise:"),
                         sg.Input(key='-NOISE PEARSON-',
                                  default_text=0.0,
                                  size=(15, 1),
                                  readonly=True),
                         sg.Text("Weighted Pearson score - noise:"),
                         sg.Input(key='-NOISE WEIGHT PEARSON-',
                                  default_text=0.0,
                                  size=(15, 1),
                                  readonly=True),                         
                     ],
                             [
                                 sg.Column([[sg.Text("BF")],
                                            [
                                                sg.Image(key='-BF-',
                                                         enable_events=True)
                                            ]]),
                                 sg.Column([[sg.Text("GT")],
                                            [
                                                sg.Image(key='-GT-',
                                                         enable_events=True)
                                            ]]),
                                 sg.Column([[sg.Text("PREDICTION")],
                                            [
                                                sg.Image(key='-PR-',
                                                         enable_events=True)
                                            ]]),
                                 sg.Column([[sg.Text("INTERPERTABILITY")],
                                            [
                                                sg.Image(key='-GC-',
                                                         enable_events=True)
                                            ]]),
                                 sg.Column([[sg.Text("NOISE PREDICTION")],
                                            [
                                                sg.Image(key='-NP-',
                                                         enable_events=True)
                                            ]])                                 
                             ]],
                     expand_x=True,
                     expand_y=True)
        ]
    ]
    return images_layout


def create_images_window():
    global images_window
    if (images_window is None):
        images_window = sg.Window('IMAGES', [create_images_layout()],
                                  finalize=True)
        images_window.bind("<Right>", "Show + Right")
        images_window.bind("<Left>", "Show + Left")
        images_window.bind("<Return>", "Show")


main_window = sg.Window('Interpertability 3D GUI', [upper_layout, middle_layout],
                        finalize=True)
main_window.bind("<Return>", "Show")  #bind the enter key to show

images_window = None
panels_window = None

mode_dict = {
    '-RADIO PIXEL-': "pixel",
    '-RADIO RECT-': "subset",
    '-RADIO FULL-': "full"
}
args = None
unet_model = None
mg_model = None
dataset_path = None
image_index = None
slice = None
layers = None
selected_layer = None

mask_norm = None

signal = None
target = None
prediction = None
mask = None
dna = None
membrane = None
mem_seg_image = None
target_seg_image = None
target_seg_image_dilated = None
noise_prediction = None

signal_mm = None
target_mm = None
prediction_mm = None
dna_mm = None
membrane_mm = None
noise_prediction_mm = None

sliced_signal = None
sliced_target = None
sliced_prediction = None
sliced_mask = None
sliced_dna = None
sliced_membrane = None
sliced_noise_prediction = None

sliced_signal_s = None
sliced_target_s = None
sliced_prediction_s = None
sliced_mask_s = None
sliced_noise_prediction_s = None

roi_mode = "subset"
roi_args = None

start_x = None
start_y = None
end_x = None
end_y = None

main_image_ndarray = None
scale_percent = 0.33

history = {}
history_image = None
sliced_history_image = None
history_key = None
history_roi_args = None
history_roi_mode = None
current_history_str = None

a = 0.5  # weight for dna and membrane
th_a = 0.5
th_b = 0.5
gamma = 0.0


def show_all(is_nuc_on, is_mem_on):
    global sliced_signal, sliced_target, sliced_prediction, sliced_mask, sliced_membrane, sliced_dna, sliced_noise_prediction, sliced_signal_s, sliced_target_s, sliced_prediction_s, sliced_mask_s, sliced_noise_prediction_s

    sliced_signal = signal_mm[slice, :, :]
    sliced_target = target_mm[slice, :, :]
    sliced_prediction = prediction_mm[slice, :, :]
 
    score = pearson_corr(np.expand_dims(sliced_target,axis=-1), np.expand_dims(sliced_prediction,axis=-1))
    images_window['-PEARSON-'].update(score)
    weighted_score = pearson_corr(np.expand_dims(sliced_target,axis=-1), np.expand_dims(sliced_prediction,axis=-1),np.expand_dims(target_seg_image_dilated[slice, :, :],axis=-1))
    images_window['-WEIGHT PEARSON-'].update(weighted_score)
    
    if noise_prediction is not None:
        try:
            roi = get_roi(roi_mode, roi_args, signal)
            sliced_noise_prediction = noise_prediction_mm[slice, :, :]
            noise_score = pearson_corr(np.expand_dims(sliced_prediction*roi.roi,axis=-1), np.expand_dims(sliced_noise_prediction*roi.roi,axis=-1))
            images_window['-NOISE PEARSON-'].update(noise_score)
            noise_weighted_score = pearson_corr(np.expand_dims(sliced_prediction*roi.roi,axis=-1), np.expand_dims(sliced_noise_prediction*roi.roi,axis=-1),np.expand_dims(target_seg_image_dilated[slice, :, :]*roi.roi,axis=-1))
            images_window['-NOISE WEIGHT PEARSON-'].update(noise_weighted_score)   
        except Exception as e:
            print("noise-pearson can not be calculated")
        
    if (mask is not None):
        sliced_mask = mask[slice, :, :].astype(np.float32)

    if (is_nuc_on):
        sliced_dna = dna_mm[slice, :, :]
        sliced_dna = np.where(sliced_dna < th_a, 0, sliced_dna)

        sliced_signal = cv2.addWeighted(
            np.uint8(
                np.dstack([sliced_signal, sliced_signal, sliced_signal]) *
                255), a,
            np.uint8(
                np.dstack([
                    np.where(sliced_dna < th_a, sliced_signal, sliced_dna),
                    np.where(sliced_dna < th_a, sliced_signal, 0),
                    np.where(sliced_dna < th_a, sliced_signal, 0)
                ]) * 255), 1 - a, gamma)
        sliced_target = cv2.addWeighted(
            np.uint8(
                np.dstack([sliced_target, sliced_target, sliced_target]) *
                255), a,
            np.uint8(
                np.dstack([
                    np.where(sliced_dna < th_a, sliced_target, sliced_dna),
                    np.where(sliced_dna < th_a, sliced_target, 0),
                    np.where(sliced_dna < th_a, sliced_target, 0)
                ]) * 255), 1 - a, gamma)
        sliced_prediction = cv2.addWeighted(
            np.uint8(
                np.dstack([
                    sliced_prediction, sliced_prediction, sliced_prediction
                ]) * 255), a,
            np.uint8(
                np.dstack([
                    np.where(sliced_dna < th_a, sliced_prediction, sliced_dna),
                    np.where(sliced_dna < th_a, sliced_prediction, 0),
                    np.where(sliced_dna < th_a, sliced_prediction, 0)
                ]) * 255), 1 - a, gamma)
        if (mask is not None):
            sliced_mask = cv2.addWeighted(
                sliced_mask, a,
                np.uint8(
                    np.dstack([
                        np.where(sliced_dna < th_a, sliced_mask[:, :, 0],
                                 sliced_dna),
                        np.where(sliced_dna < th_a, sliced_mask[:, :, 1], 0),
                        np.where(sliced_dna < th_a, sliced_mask[:, :, 2], 0)
                    ])), 1 - a, gamma)

    if (is_mem_on and not is_nuc_on):
        sliced_membrane = membrane_mm[slice, :, :]
        sliced_membrane = np.where(sliced_membrane < th_b, 0, sliced_membrane)

        sliced_signal = cv2.addWeighted(
            np.uint8(
                np.dstack([sliced_signal, sliced_signal, sliced_signal]) *
                255), a,
            np.uint8(
                np.dstack([
                    np.where(sliced_membrane < th_b, sliced_signal, 0),
                    np.where(sliced_membrane < th_b, sliced_signal,
                             sliced_membrane),
                    np.where(sliced_membrane < th_b, sliced_signal, 0)
                ]) * 255), 1 - a, gamma)
        sliced_target = cv2.addWeighted(
            np.uint8(
                np.dstack([sliced_target, sliced_target, sliced_target]) *
                255), a,
            np.uint8(
                np.dstack([
                    np.where(sliced_membrane < th_b, sliced_target, 0),
                    np.where(sliced_membrane < th_b, sliced_target,
                             sliced_membrane),
                    np.where(sliced_membrane < th_b, sliced_target, 0)
                ]) * 255), 1 - a, gamma)
        sliced_prediction = cv2.addWeighted(
            np.uint8(
                np.dstack([
                    sliced_prediction, sliced_prediction, sliced_prediction
                ]) * 255), a,
            np.uint8(
                np.dstack([
                    np.where(sliced_membrane < th_b, sliced_prediction, 0),
                    np.where(sliced_membrane < th_b, sliced_prediction,
                             sliced_membrane),
                    np.where(sliced_membrane < th_b, sliced_prediction, 0)
                ]) * 255), 1 - a, gamma)
        if (mask is not None):
            sliced_mask = cv2.addWeighted(
                sliced_mask, a,
                np.uint8(
                    np.dstack([
                        np.where(sliced_membrane < th_b, sliced_mask[:, :, 0],
                                 0),
                        np.where(sliced_membrane < th_b, sliced_mask[:, :, 1],
                                 sliced_membrane),
                        np.where(sliced_membrane < th_b, sliced_mask[:, :, 2],
                                 0)
                    ])), 1 - a, gamma)

    elif (is_mem_on and is_nuc_on):
        sliced_membrane = membrane_mm[slice, :, :]
        sliced_membrane = np.uint8(
            np.where(sliced_membrane < th_b, 0, sliced_membrane) * 255)

        sliced_signal = cv2.addWeighted(
            sliced_signal, a,
            np.uint8(
                np.dstack([
                    np.where(sliced_membrane < th_b, sliced_signal[:,:,:1],
                             0),
                    np.where(sliced_membrane < th_b,  sliced_signal[:,:,1:2],
                             sliced_membrane),
                    np.where(sliced_membrane < th_b,  sliced_signal[:,:,2:3], 0)
                ])), 1 - a, gamma)

        sliced_target = cv2.addWeighted(
            sliced_target, a,
            np.uint8(
                np.dstack([
                    np.where(sliced_membrane < th_b, sliced_target[:,:,:1],
                             0),
                    np.where(sliced_membrane < th_b, sliced_target[:,:,1:2],
                             sliced_membrane),
                    np.where(sliced_membrane < th_b, sliced_target[:,:,2:3], 0)
                ])), 1 - a, gamma)

        sliced_prediction = cv2.addWeighted(
            sliced_prediction, a,
            np.uint8(
                np.dstack([
                    np.where(sliced_membrane < th_b, sliced_prediction[:,:,:1], 0),
                    np.where(sliced_membrane < th_b,
                             sliced_prediction[:,:,1:2], sliced_membrane),
                    np.where(sliced_membrane < th_b, sliced_prediction[:,:,2:3], 0)
                ])), 1 - a, gamma)
        if (mask is not None):
            sliced_mask = cv2.addWeighted(
                sliced_mask, a,
                np.uint8(
                    np.dstack([
                        np.where(sliced_membrane < th_b, sliced_mask[:, :, :1],
                                 0),
                        np.where(sliced_membrane < th_b, sliced_mask[:, :, 1:2],
                                 sliced_membrane),
                        np.where(sliced_membrane < th_b, sliced_mask[:, :, 2:3],
                                 0)
                    ])), 1 - a, gamma)

    width = int(sliced_signal.shape[1] * scale_percent)
    height = int(sliced_signal.shape[0] * scale_percent)
    dim = (width, height)

    sliced_signal_s = cv2.resize(sliced_signal,
                                 dim,
                                 interpolation=cv2.INTER_AREA)
    sliced_target_s = cv2.resize(sliced_target,
                                 dim,
                                 interpolation=cv2.INTER_AREA)
    sliced_prediction_s = cv2.resize(sliced_prediction,
                                     dim,
                                     interpolation=cv2.INTER_AREA)
    if (sliced_mask is not None):
        sliced_mask_s = cv2.resize(sliced_mask,
                                   dim,
                                   interpolation=cv2.INTER_AREA)
        
    if (sliced_noise_prediction is not None):
        sliced_noise_prediction_s = cv2.resize(sliced_noise_prediction,
                                   dim,
                                   interpolation=cv2.INTER_AREA)        

    show_image(sliced_signal_s, '-BF-')
    show_image(sliced_target_s, '-GT-')
    show_image(sliced_prediction_s, '-PR-')
    if (sliced_mask is not None):
        show_image(sliced_mask_s, '-GC-')
    if (sliced_noise_prediction is not None):
        show_image(sliced_noise_prediction_s, '-NP-')        

def binds_all():
    # Mouse events bind
    images_window['-BF-'].Widget.bind(
        "<Button-1>",
        lambda event: on_image_left_click(event, sliced_signal_s, '-BF-'))
    images_window['-GT-'].Widget.bind(
        "<Button-1>",
        lambda event: on_image_left_click(event, sliced_target_s, '-GT-'))
    images_window['-PR-'].Widget.bind(
        "<Button-1>",
        lambda event: on_image_left_click(event, sliced_prediction_s, '-PR-'))
    images_window['-GC-'].Widget.bind(
        "<Button-1>",
        lambda event: on_image_left_click(event, sliced_mask_s, '-GC-'))
    images_window['-NP-'].Widget.bind(
        "<Button-1>",
        lambda event: on_image_left_click(event, sliced_noise_prediction_s, '-NP-'))    
    
    images_window['-MAIN IMAGE-'].Widget.bind(
        "<Button-1>", lambda event: on_image_left_click(
            event, main_image_ndarray, '-MAIN IMAGE-', scale_percent))

    images_window['-BF-'].Widget.bind(
        "<B1-Motion>",
        lambda event: on_image_left_motion(event, sliced_signal_s, '-BF-'))
    images_window['-GT-'].Widget.bind(
        "<B1-Motion>",
        lambda event: on_image_left_motion(event, sliced_target_s, '-GT-'))
    images_window['-PR-'].Widget.bind(
        "<B1-Motion>",
        lambda event: on_image_left_motion(event, sliced_prediction_s, '-PR-'))
    images_window['-GC-'].Widget.bind(
        "<B1-Motion>",
        lambda event: on_image_left_motion(event, sliced_mask_s, '-GC-'))
    images_window['-NP-'].Widget.bind(
        "<B1-Motion>",
        lambda event: on_image_left_motion(event, sliced_noise_prediction_s, '-NP-'))    
    
    images_window['-MAIN IMAGE-'].Widget.bind(
        "<B1-Motion>", lambda event: on_image_left_motion(
            event, main_image_ndarray, '-MAIN IMAGE-', scale_percent))

    images_window['-BF-'].Widget.bind(
        "<Button-3>", lambda event: on_image_right_click(event, sliced_signal))
    images_window['-GT-'].Widget.bind(
        "<Button-3>", lambda event: on_image_right_click(event, sliced_target))
    images_window['-PR-'].Widget.bind(
        "<Button-3>",
        lambda event: on_image_right_click(event, sliced_prediction))
    images_window['-GC-'].Widget.bind(
        "<Button-3>", lambda event: on_image_right_click(event, sliced_mask))
    images_window['-NP-'].Widget.bind(
        "<Button-3>", lambda event: on_image_right_click(event, sliced_noise_prediction))    

    # Opposite draw on the others
    images_window['-GT-'].Widget.bind(
        "<ButtonRelease-1>",
        lambda event: on_image_left_release(event, [
            sliced_signal_s, sliced_target_s, sliced_prediction_s,
            sliced_mask_s,sliced_noise_prediction_s
        ], ['-BF-', '-GT-', '-PR-', '-GC-','-NP-'], 1))
    images_window['-BF-'].Widget.bind(
        "<ButtonRelease-1>",
        lambda event: on_image_left_release(event, [
            sliced_signal_s, sliced_target_s, sliced_prediction_s,
            sliced_mask_s,sliced_noise_prediction_s
        ], ['-BF-', '-GT-', '-PR-', '-GC-','-NP-'], 1))
    images_window['-PR-'].Widget.bind(
        "<ButtonRelease-1>",
        lambda event: on_image_left_release(event, [
            sliced_signal_s, sliced_target_s, sliced_prediction_s,
            sliced_mask_s, sliced_noise_prediction_s
        ], ['-BF-', '-GT-', '-PR-', '-GC-','-NP-'], 1))
    images_window['-GC-'].Widget.bind(
        "<ButtonRelease-1>",
        lambda event: on_image_left_release(event, [
            sliced_signal_s, sliced_target_s, sliced_prediction_s,
            sliced_mask_s, sliced_noise_prediction_s
        ], ['-BF-', '-GT-', '-PR-', '-GC-','-NP-'], 1))
    images_window['-MAIN IMAGE-'].Widget.bind(
        "<ButtonRelease-1>",
        lambda event: on_image_left_release(event, [
            sliced_signal_s, sliced_target_s, sliced_prediction_s,
            sliced_mask_s, sliced_noise_prediction_s
        ], ['-BF-', '-GT-', '-PR-', '-GC-','-NP-'], scale_percent))

def show_image(image_ndarray, key, window=images_window):
    if (window is None):
        window = images_window
    if (image_ndarray is not None):
        if (len(image_ndarray.shape) == 2) or (image_ndarray.shape[2] == 1):  # not for mask
            cv_image = np.uint8(
                np.dstack([image_ndarray, image_ndarray, image_ndarray]) * 255)
        else:
            cv_image = image_ndarray.copy().astype(np.uint8)
        image = Image.fromarray(cv_image).convert("RGB")
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        window[key].update(data=bio.getvalue())

def add_rect(image_ndarray, start, end, key):
    if (image_ndarray is not None):
        if (len(image_ndarray.shape) == 2):  # not for mask
            cv_image = np.uint8(
                np.dstack([image_ndarray, image_ndarray, image_ndarray]) * 255)
        else:
            cv_image = image_ndarray.copy()
        cv_image = cv2.rectangle(cv_image, start, end, (0, 0, 255), 2)
        show_image(cv_image, key)

def add_circle(image_ndarray, start, key):
    if (image_ndarray is not None):
        if (len(image_ndarray.shape) == 2):  # not for mask
            cv_image = np.uint8(
                np.dstack([image_ndarray, image_ndarray, image_ndarray]) * 255)
        else:
            cv_image = image_ndarray.copy()
        cv_image = cv2.circle(cv_image, start, 1, (0, 0, 255), 3)
        show_image(cv_image, key)

def on_image_left_click(event, image_ndarray, key, scale=1):
    if (image_ndarray is not None):
        global start_x, start_y
        start_x = int(event.x * scale)
        start_y = int(event.y * scale)
        if (roi_mode == "pixel"):
            add_circle(image_ndarray,
                       (int(start_x / scale), int(start_y / scale)), key)

def on_image_right_click(event, image_ndarray):
    global main_image_ndarray
    images_window['-STATUS-'].update(history_key)
    main_image_ndarray = image_ndarray
    if (image_ndarray is not None):
        key = '-MAIN IMAGE-'
        show_image(image_ndarray, key)

        # Redraw roi
        if (roi_mode == "pixel"):
            if (start_x is not None and start_y is not None):
                add_circle(image_ndarray, (int(
                    start_x / scale_percent), int(start_y / scale_percent)),
                           key)
        elif (roi_mode == "subset"):
            if (start_x is not None and start_y is not None
                    and end_x is not None and end_y is not None):
                add_rect(
                    image_ndarray, (int(start_x / scale_percent),
                                    int(start_y / scale_percent)),
                    (int(end_x / scale_percent), int(end_y / scale_percent)),
                    key)

def on_image_left_motion(event, image_ndarray, key, scale=1):
    if (image_ndarray is not None):
        motion_x = event.x
        motion_y = event.y
        if (roi_mode == "subset"):
            add_rect(image_ndarray,
                     (int(start_x / scale), int(start_y / scale)),
                     (motion_x, motion_y), key)
            main_window['-ROI TEXT-'].update("({},{}),({},{}),slice:{}".format(
                start_x, start_y, motion_x, motion_y, slice))

def on_image_left_release(event, images_ndarray, keys, scale=1):
    global end_x, end_y, roi_args
    end_x = int(event.x * scale)
    end_y = int(event.y * scale)
    for i in range(len(images_ndarray)):
        image_ndarray = images_ndarray[i]
        if (image_ndarray is not None):
            key = keys[i]
            if (roi_mode == "pixel"):
                add_circle(image_ndarray, (start_x, start_y), key)
                main_window['-ROI TEXT-'].update("({},{}),slice:{}".format(
                    start_x, start_y, slice))
                roi_args = [
                    int(start_x / scale_percent),
                    int(start_y / scale_percent), slice
                ]
            elif (roi_mode == "subset"):
                add_rect(image_ndarray, (start_x, start_y), (end_x, end_y),
                         key)
                main_window['-ROI TEXT-'].update(
                    "({},{}),({},{}),slice:{}".format(start_x, start_y, end_x,
                                                      end_y, slice))
                roi_args = [
                    int(start_x / scale_percent),
                    int(start_y / scale_percent),
                    int(end_x / scale_percent),
                    int(end_y / scale_percent), slice
                ]
            elif (roi_mode == "full"):
                main_window['-ROI TEXT-'].update("slice:{}".format(slice))
                roi_args = [slice]

    if (main_image_ndarray is not None):
        key = '-MAIN IMAGE-'
        if (roi_mode == "pixel"):
            add_circle(
                main_image_ndarray,
                (int(start_x / scale_percent), int(start_y / scale_percent)),
                key)
        elif (roi_mode == "subset"):
            add_rect(
                main_image_ndarray,
                (int(start_x / scale_percent), int(start_y / scale_percent)),
                (int(end_x / scale_percent), int(end_y / scale_percent)), key)
    main_window['Calculate'].update(disabled=False)

def redraw_roi():
    # Redraw roi
    if (roi_mode == "pixel"):
        if (start_x is not None and start_y is not None):
            add_circle(sliced_signal_s, (start_x, start_y), '-BF-')
            add_circle(sliced_target_s, (start_x, start_y), '-GT-')
            add_circle(sliced_prediction_s, (start_x, start_y), '-PR-')
            add_circle(sliced_mask_s, (start_x, start_y), '-GC-')
            add_circle(sliced_noise_prediction_s, (start_x, start_y), '-NP-')
    elif (roi_mode == "subset"):
        if (start_x is not None and start_y is not None and end_x is not None
                and end_y is not None):
            add_rect(sliced_signal_s, (start_x, start_y), (end_x, end_y),
                     '-BF-')
            add_rect(sliced_target_s, (start_x, start_y), (end_x, end_y),
                     '-GT-')
            add_rect(sliced_prediction_s, (start_x, start_y), (end_x, end_y),
                     '-PR-')
            add_rect(sliced_mask_s, (start_x, start_y), (end_x, end_y), '-GC-')
            add_rect(sliced_noise_prediction_s, (start_x, start_y), (end_x, end_y), '-NP-')

# Display and interact with the Window using an Event Loop
while True:
    window, event, values = sg.read_all_windows()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED:
        if window == images_window:
            window.close()
            images_window = None
        if window == panels_window:
            window.close()
            panels_window = None
        elif window == main_window:
            break

    # Load model, load dataset, run predictions and gatther data to gui
    elif event == 'Load':
        is_valid = True

        # Check model and load it
        if (main_window.ReturnValuesDictionary['-UNET MODEL-'] != ''
                and main_window.ReturnValuesDictionary['-UNET MODEL-'] is not None):
            unet_model = load_model(main_window.ReturnValuesDictionary['-UNET MODEL-'])
        else:
            is_valid = False
            sg.popup_error("UNET Model value is incorrect, {}".format(
                main_window.ReturnValuesDictionary['-UNET MODEL-']))
        
        # Check model and load it
        if (main_window.ReturnValuesDictionary['-MI MODEL-'] != ''
                and main_window.ReturnValuesDictionary['-MI MODEL-'] is not None):
            mg_model = load_model(main_window.ReturnValuesDictionary['-MI MODEL-'])
        else:
            is_valid = False
            sg.popup_error("Mask Interpreter Model value is incorrect, {}".format(
                main_window.ReturnValuesDictionary['-MI MODEL-']))        

        # Check dataset path
        if (main_window.ReturnValuesDictionary['-DATASET-'] != '' and
                main_window.ReturnValuesDictionary['-DATASET-'] is not None):
            dataset_path = main_window.ReturnValuesDictionary['-DATASET-']
        else:
            is_valid = False
            sg.popup_error("Dataset value is incorrect, {}".format(
                main_window.ReturnValuesDictionary['-DATASET-']))

        if (is_valid):
            dataset = get_dataset(dataset_path)
            layers = collect_layers(unet_model)
            
            signal, target, target_seg_image, dna, membrane, mem_seg_image = item_from_dataset(dataset, 0)
            # Update slider
            main_window['-SLICE-'].update(range=(0, signal.shape[0] - 1))
            # Enable show button
            main_window['Show'].update(disabled=False)
            # Add data to layers combo
            main_window['-LAYERS-'].update(values=list(layers.keys()))
            main_window['-LAYERS-'].update(value=list(layers.keys())[0])
            selected_layer = layers[list(layers.keys())[0]]


            signal = None
            target = None
            prediction = None
            mask = None
            dna = None
            membrane = None
            mem_seg_image = None
            target_seg_image = None
            mask = None
            if (images_window is not None):
                images_window.close()
                
            main_window['Evaluate interperters'].update(disabled=False)
    
    elif event == 'Plot pearson score by slice':
        scores = get_pearson_per_slice(prediction,target)
        plot_pearson_per_slice(scores, "Organelle")
    
    elif event == "Plot weighted pearson score by slice":
        scores = get_pearson_per_slice(prediction,target,target_seg_image_dilated)
        plot_pearson_per_slice(scores, "Organelle")        
        
        
    elif 'Show' in event and event != 'Show History':

        is_valid = True
        is_index_changed = False

        if ('Right' in event):
            temp_v = np.maximum(
                0,
                np.minimum(
                    signal.shape[1] - 1,
                    int(main_window.ReturnValuesDictionary['-SLICE-']) + 1))
            main_window.ReturnValuesDictionary['-SLICE-'] = temp_v
            main_window['-SLICE-'].update(value=temp_v)
        if ('Left' in event):
            temp_v = np.maximum(
                0,
                np.minimum(
                    signal.shape[1] - 1,
                    int(main_window.ReturnValuesDictionary['-SLICE-']) - 1))
            main_window.ReturnValuesDictionary['-SLICE-'] = temp_v
            main_window['-SLICE-'].update(value=temp_v)

        if (images_window is not None):
            dna_th = float(images_window.ReturnValuesDictionary['-DNA TH-'])
            if (dna_th >= 0 and dna_th <= 1):
                th_a = dna_th
            mem_th = float(images_window.ReturnValuesDictionary['-MEM TH-'])
            if (mem_th >= 0 and mem_th <= 1):
                th_b = mem_th
            a = float(images_window.ReturnValuesDictionary['-GAM TH-'])

        # Load image from dataset
        if (main_window.ReturnValuesDictionary['-IMAGE INDEX-'] != '' and
            (main_window.ReturnValuesDictionary['-IMAGE INDEX-'] is not None)
                and
                int(main_window.ReturnValuesDictionary['-IMAGE INDEX-']) >= 0):
            if (int(main_window.ReturnValuesDictionary['-IMAGE INDEX-']) !=
                    image_index or
                (signal is None or target is None or prediction is None)):
                image_index = int(
                    main_window.ReturnValuesDictionary['-IMAGE INDEX-'])
                signal, target, target_seg_image, dna, membrane, mem_seg_image = item_from_dataset(
                    dataset, image_index)
                target_seg_image_dilated = np.copy(target_seg_image)
                for h in range(target_seg_image.shape[1]):
                    target_seg_image_dilated[0, h, :, :] = cv2.dilate(target_seg_image_dilated[0, h, :, :].astype(np.uint8), np.ones((17,17)))                  
                prediction = predict(unet_model, signal)
                
                signal_mm = ImageUtils.normalize(signal,1.0,np.float32)
                target_mm =  ImageUtils.normalize(target,1.0,np.float32)
                dna_mm =  ImageUtils.normalize(dna,1.0,np.float32)
                membrane_mm =  ImageUtils.normalize(membrane,1.0,np.float32)
                prediction_mm =  ImageUtils.normalize(prediction,1.0,np.float32)
                is_index_changed = True

        else:
            is_valid = False
            sg.popup_error(
                "Image Index value is incorrect, {}, should be integer".format(
                    main_window.ReturnValuesDictionary['-IMAGE INDEX-']))

        if (main_window.ReturnValuesDictionary['-SLICE-'] != ''
                and (main_window.ReturnValuesDictionary['-SLICE-'] is not None)
                and int(main_window.ReturnValuesDictionary['-SLICE-']) >= 0
                and int(main_window.ReturnValuesDictionary['-SLICE-']) <
                signal.shape[0]):
            if (int(main_window.ReturnValuesDictionary['-SLICE-']) != slice
                    or is_index_changed):
                slice = int(main_window.ReturnValuesDictionary['-SLICE-'])

        else:
            is_valid = False
            sg.popup_error(
                "Slice value is incorrect, {}, should be integer".format(
                    main_window.ReturnValuesDictionary['-SLICE-']))

        if (is_valid):
            create_images_window()
            show_all(images_window.ReturnValuesDictionary['-NUC CHECK-'],
                     images_window.ReturnValuesDictionary['-MEM CHECK-'])

            binds_all()

            # enable checkboxs
            if (dna is not None):
                images_window['-NUC CHECK-'].update(disabled=False)

            if (membrane is not None):
                images_window['-MEM CHECK-'].update(disabled=False)

            redraw_roi()
            images_window['-STATUS-'].update("")

    elif (event == '-RADIO PIXEL-' or event == '-RADIO RECT-'
          or event == '-RADIO FULL-'):
        if (roi_mode != mode_dict[event]):

            # Mode changed, reset image
            roi_mode = mode_dict[event]
            roi_args = [slice]
            if ((signal is not None) and (target is not None)):
                start_x = None
                start_y = None
                end_x = None
                end_y = None

                show_all(images_window.ReturnValuesDictionary['-NUC CHECK-'],
                         images_window.ReturnValuesDictionary['-MEM CHECK-'])

                main_window['-ROI TEXT-'].update('')
                main_window['Noise prediction'].update(disabled=True)
                
                if (roi_mode == "full"):
                    main_window['Calculate'].update(disabled=False)
                else:
                    main_window['Calculate'].update(disabled=True)
                    

    elif (event == '-LAYERS-'):
        if (main_window.ReturnValuesDictionary['-LAYERS-'] != '' and
                main_window.ReturnValuesDictionary['-LAYERS-'] is not None):
            selected_layer = layers[
                main_window.ReturnValuesDictionary['-LAYERS-']]

    elif (event == 'Calculate'):

        # Add prev mask to history
        if (history_image is not None and history_key is not None):
            history.update({
                history_key: {
                    "image": history_image,
                    "roi_mode": history_roi_mode,
                    "roi_args": history_roi_args
                }
            })
            images_window['-HISTORY-'].update(values=list(history.keys()))
            images_window['-HISTORY-'].update(value=current_history_str)
            images_window['Show History'].update(disabled=False)
        
        method = ""
        interperter = None
        if main_window.ReturnValuesDictionary['-RADIO SM-']:
            method="saliency"
            interperter = Saliency(unet_model)
            history_key = "{}-{}-{}-{}".format(method,image_index,roi_mode, str(roi_args))
        elif main_window.ReturnValuesDictionary['-RADIO GBP-']:
            method="gbp"
            interperter = GuidedBackprop(unet_model)
            history_key = "{}-{}-{}-{}".format(method,image_index,roi_mode, str(roi_args))
        elif main_window.ReturnValuesDictionary['-RADIO GC-']:
            X_gradcam = main_window.ReturnValuesDictionary['-X gradcam-']
            if X_gradcam:
                method = "X_gradcam"
            else:
                method = "gradcam"
            interperter = GradCam(model=unet_model,
                    target_layer=selected_layer,
                    X_gradcam=X_gradcam)
            
            history_key = "{}-{}-{}-{}-{}".format(method,image_index, values['-LAYERS-'],roi_mode, str(roi_args))
        elif main_window.ReturnValuesDictionary['-RADIO MI-']:
            method="mask_interperter"
            interperter = MaskInterperter(model=mg_model)
            history_key = "{}-{}-{}-{}".format(method,image_index,roi_mode, str(roi_args))

        mask, mask_norm = get_mask(interperter, signal, roi_mode, roi_args)

        save_mask(gv.mg_model_path, mask_norm, selected_layer.name, roi_mode, roi_args, method)     
        
        noise_prediction = None
        noise_prediction_mm = None   

        show_all(images_window.ReturnValuesDictionary['-NUC CHECK-'],
                 images_window.ReturnValuesDictionary['-MEM CHECK-'])
        redraw_roi()
        
        main_window['Noise prediction'].update(disabled=False)

        history_image = mask.copy()
        if (roi_args is not None):
            history_roi_args = roi_args.copy()
        else:
            history_roi_args = None
        history_roi_mode = roi_mode
        
    elif (event == 'Noise prediction'):
        mask_th = float(main_window.ReturnValuesDictionary['-NP TH-'])
        noise_prediction = get_noise_prediction(signal,mask_norm, mask_th, unet_model, roi_mode, roi_args)
        noise_prediction_mm = noise_prediction
        show_all(images_window.ReturnValuesDictionary['-NUC CHECK-'],images_window.ReturnValuesDictionary['-MEM CHECK-'])
        redraw_roi()
    
    elif (event == 'Evaluate interperters'):
        X_gradcam = main_window.ReturnValuesDictionary['-X gradcam-']
        evaluate_interperters(gv.mg_model_path,dataset,unet_model,mg_model,selected_layer,X_gradcam)    
        
    elif (event == 'Show History'):
        if (images_window.ReturnValuesDictionary['-HISTORY-'] != '' and
                images_window.ReturnValuesDictionary['-HISTORY-'] is not None):
            current_history_str = images_window.ReturnValuesDictionary[
                '-HISTORY-']
            current_history = history[
                images_window.ReturnValuesDictionary['-HISTORY-']]
            sliced_history_image = current_history["image"][slice, :, :]
            show_image(sliced_history_image, '-HISTORY IMAGE-')

            # Redraw roi
            if (current_history["roi_mode"] == "pixel"):
                if (current_history["roi_args"] is not None):
                    add_circle(sliced_history_image,
                               (current_history["roi_args"][0],
                                current_history["roi_args"][1]),
                               '-HISTORY IMAGE-')
            elif (current_history["roi_mode"] == "subset"):
                if (current_history["roi_args"] is not None):
                    add_rect(sliced_history_image,
                             (current_history["roi_args"][0],
                              current_history["roi_args"][1]),
                             (current_history["roi_args"][2],
                              current_history["roi_args"][3]),
                             '-HISTORY IMAGE-')

    elif (event == '-NUC CHECK-' or event == '-MEM CHECK-'):
        dna_th = float(images_window.ReturnValuesDictionary['-DNA TH-'])
        if (dna_th >= 0 and dna_th <= 1):
            th_a = dna_th
        mem_th = float(images_window.ReturnValuesDictionary['-MEM TH-'])
        if (mem_th >= 0 and mem_th <= 1):
            th_b = mem_th
        a = float(images_window.ReturnValuesDictionary['-GAM TH-'])
        show_all(images_window.ReturnValuesDictionary['-NUC CHECK-'],
                 images_window.ReturnValuesDictionary['-MEM CHECK-'])
        redraw_roi()

# Finish up by removing from the screen
main_window.close()