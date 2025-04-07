# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:10:08 2025

@author: diego
"""


import cv2
import numpy as np


def read_video_frames_and_format(video_path: str, size: tuple, target_frames: int):
    """
    Reads a video using opencv and then:
        Resixe the frames to be the specified size (n, m)
        Gets exactly the number of frames specify
        Returns an array with all the frames, shape: (number_frames, n, m, 3)
    """
    

    # read it 
    video = cv2.VideoCapture(video_path)

    # get original video info
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Original video:")
    print(f"Total number of frames: {total_frames}")
    print(f"Frame shape: ({frame_height}, {frame_width})")

    # save all frames in a list
    frames = []

    # resize all frames
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames.append(frame)
    video.release()


    # normalize frame count
    current_frame_count = len(frames)
    if current_frame_count > target_frames:
        # sample from the current frames, evenly spaced
        step = current_frame_count / target_frames
        frames = [frames[int(i * step)] for i in range(target_frames)]
    elif current_frame_count < target_frames:
        # duplicate to reach the desired number
        indices = np.linspace(0, current_frame_count - 1, target_frames).astype(int)
        frames = [frames[i] for i in indices]

    # transfor to array of shape (number_of_frames, n, m, 3)
    frames = np.array(frames)  
        
    # check new info
    print(f"\nNew frames shape: {frames.shape}")
    print("-"*50)

    return frames



def transform_frames_2_grayscale(video_frames):
    """
    Given the frames of a video:  (number_frames, n, m, 3)
    transform them to grayscale:  (number_frames, n, m)
    """
    
    print(f"Original shape: {video_frames.shape}")
    
    bn_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                for frame in video_frames]
    bn_frames = np.array(bn_frames)
    
    print(f"Grayscale shape: {bn_frames.shape}")
    print("-"*50)
    
    return bn_frames



def process_frames_video(video_frames):
    """ 
    Process the frames of a video
    Hace que la esperanza sea 0 y la varianza 1 de cada frame
    """
    
    return (video_frames - video_frames.mean())/video_frames.std()



def mezclar_videos(A, videos_arr):
    """ 
    Dado un vector con los videos, forma: (num_videos, num_frames_video, N, N, color_chanels)
    y una matriz A sxs de mezcla (hay s videos)
    obtiene 4 videos mezcla, 
    donde cada frame de un video mezcla es una mezcla de frames de los videos originales
    """
    
    print(f"Shape original: {videos_arr.shape}")
    
    # ver cuantos videos hay
    s = videos_arr.shape[0]

    # almacenar los frames mezclados
    mezclas = [] 

    # iterar por cada frame
    for i in range(videos_arr.shape[1]): 

        # tomar el i-esimo frame de cada video
        frame_vector = videos_arr[:, i] 

        # obtener la mezcla de estos 4 frames
        mezcla_frames = [sum(A[i][j] * frame_vector[j] for j in range(s))
                         for i in range(s)]
        mezcla_frames = np.array(mezcla_frames) # shape (num_videos, N, N, color_chanels)

        # añadir esta mezcla
        mezclas.append(mezcla_frames)

    # hacer array y poner en formato correcto
    mezclas_arr = np.array(mezclas)
    # considerar si es que tiene o no tiene canales de color
    # Si tiene 4 dimensiones (grayscale), transponer (1, 0, 2, 3)
    # Si tiene 5 dimensiones (color), transponer (1, 0, 2, 3, 4)
    if mezclas_arr.ndim == 4:  # Sin canal de color
        mezclas_arr = np.transpose(mezclas_arr, (1, 0, 2, 3))
    elif mezclas_arr.ndim == 5:  # Con canal de color
        mezclas_arr = np.transpose(mezclas_arr, (1, 0, 2, 3, 4))
        
    print(f"Shape mixture: {mezclas_arr.shape}")
    print("-"*40)
    
    return mezclas_arr



# esta funcion si la hice con chatgpt
def concatenate_videos(video_arrays, border_size=10, output_path="output.mp4", fps=30):
    """
    Une los videos en una cuadrícula de 3x4 con un borde negro entre ellos.
    
    Args:
        video_arrays (list of np.array): Lista de arrays (3 elementos), cada uno de forma (4, num_frames, N, N, 3).
        border_size (int): Tamaño del espacio negro entre los videos.
        output_path (str): Ruta donde se guardará el video de salida.
        fps (int): Cuadros por segundo del video resultante.
    """
    num_rows = len(video_arrays)  # 3 filas
    num_cols = len(video_arrays[0])  # 4 columnas
    
    num_frames = video_arrays[0][0].shape[0]  # Cantidad de frames
    N = video_arrays[0][0].shape[1]  # Tamaño de cada video (N, N)
    
    # Color negro para el borde
    black_border_v = np.zeros((N, border_size, 3), dtype=np.uint8)
    black_border_h = np.zeros((border_size, num_cols * (N + border_size) - border_size, 3), dtype=np.uint8)
    
    # Configurar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                          (num_cols * (N + border_size) - border_size, num_rows * (N + border_size) - border_size))
    
    for i in range(num_frames):
        # Concatenar las filas
        row_videos = []
        for row in range(num_rows):
            row_frames = [video_arrays[row][col][i] for col in range(num_cols)]
            row_with_borders = [np.hstack([frame, black_border_v]) for frame in row_frames]
            row_videos.append(np.hstack(row_with_borders)[:, :-border_size])  # Quitar borde extra del final
        
        # Concatenar filas con espacio en negro
        final_frame = np.vstack([row_videos[row] if row == 0 else np.vstack([black_border_h, row_videos[row]]) 
                                 for row in range(num_rows)])
        
        out.write(final_frame)
    
    out.release()
    print(f"Video guardado en {output_path}")



def procesar_videos_color(frames_videos_color):
    """
    Dados los framess de varios videos a color
    (num_videos, num_frames, N, M, 3)
    procesarlos para que sean rgb
    """
    
    frames_videos_color_rgb = []
    
    for frames_video in frames_videos_color:
        # procesar de este video
        frames_videos_color_rgb.append(procesar_video_color(frames_video))
    return np.array(frames_videos_color_rgb)



def procesar_video_color(frames_video_color):
    """
    Dados los frames de un video a color
    (num_frames, N, M, 3)
    procesarlo para que sea rgb.
    Es decir, que las entradas sean enteros de 8 bits
    """
    
    frames_video_color_0_1 = (frames_video_color - frames_video_color.min())/(frames_video_color.max() - frames_video_color.min())
    frames_video_color_0_255 = frames_video_color_0_1*255
    frames_video_color_rgb = np.round(frames_video_color_0_255).astype(np.uint8)
    
    return frames_video_color_rgb




def obtener_imagen_con_valores_video(array_video, indices_seleccionados, M):
    """
    Dado un un video, array de la forma (N_frames, N1, N2)
    y un array de indices, de la forma (M**2, 3)
    se obtienen los valores indicados por los indices,
    y se ponen en una matriz (M, M)
    """
    assert len(indices_seleccionados) == M**2
    
    # tomar todos los valores
    valores_extraidos = array_video[indices_seleccionados[:, 0], indices_seleccionados[:, 1], indices_seleccionados[:, 2]]
    # poner en matriz
    valores_extraidos = valores_extraidos.reshape(M, M)
    return valores_extraidos





