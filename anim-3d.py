import os
import shutil
from time import sleep
import numpy as np
from vispy import app, scene
from vispy.color import Color, Colormap
from vispy.visuals.transforms import STTransform

import matplotlib.pyplot as plt
from PIL import Image
import io

def latex_to_image(latex_str, dpi=500, fontsize=24):
    """Render LaTeX math text into a RGBA numpy array."""
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, latex_str, fontsize=fontsize, ha="center", va="center")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf).convert("RGBA")
    return np.array(img)

X_MIN = -20
X_MAX = 20

Y_MIN = -20
Y_MAX = 20

Z_MAX = 200

SAMPLES = 100

os.environ['VISPY_APP_BACKEND'] = 'pyqt5'

GRAPH_Z_CAMERA_OFFSET = 23
SPHERE_D = 20

def rastrigin_3D(X, Y, A=10):
    return (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

def ackley_3D(X, Y, A=20, B=0.2, C=2 * np.pi):
    term1 = -A * np.exp(-B * np.sqrt(0.5 * (X**2 + Y**2)))
    term2 = -np.exp(0.5 * (np.cos(C * X) + np.cos(C * Y)))
    return term1 + term2 + A + np.exp(1)

TRAIN_DATA_SAMPLES = 50

noise_min = -5
noise_max = 5

np.random.seed(41)

noise_xyz = np.random.uniform(noise_min, noise_max, size=(TRAIN_DATA_SAMPLES, 3))
noise_A = np.random.uniform(5, 15, size=(TRAIN_DATA_SAMPLES,))

BATCH_SIZE = 1

MAX_EPOCHS = 10
LEARNING_RATE = 0.03

current_epoch = 0

rand_index = 0 # update each epoch define epoches !

def min_fn(x, y):
    fns = [rastrigin_3D(x - nx, y - ny, nA) + nz for nx, ny, nz, nA in zip(noise_xyz[:, 0], noise_xyz[:, 1], noise_xyz[:, 2], noise_A)]

    # fns = [rastrigin_3D(x, y + 5, -5)]
    if (BATCH_SIZE < TRAIN_DATA_SAMPLES):
        # STOCHASTIC GRADIENT DESCENT
        fns = np.array(fns)[rand_index: rand_index + BATCH_SIZE]
    else:
        # GRADIENT DESCENT
        fns = np.array(fns)
    

    return np.mean(fns, axis=0)


x_0 = 15
y_0 = 15
x = x_0
y = y_0

min_z = min_fn(x_0, y_0)


def init():
    # Generate data
    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    y = np.linspace(Y_MIN, Y_MAX, SAMPLES)
    X, Y = np.meshgrid(x, y)
    Z = min_fn(X, Y)

    # Create a canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white', size=(600, 500))
    
    fixed_left_bottom_title_text = ""

    if (BATCH_SIZE < TRAIN_DATA_SAMPLES):
        fixed_left_bottom_title_text = "Mini-Batch\n Gradient\n Descent"
    
    if (BATCH_SIZE == TRAIN_DATA_SAMPLES):
        fixed_left_bottom_title_text = "Gradient\n Descent"

    if (BATCH_SIZE == 1):
        fixed_left_bottom_title_text = "Stochastic\n Gradient\n Descent"

    fixed_left_bottom_title = scene.visuals.Text(
        text=fixed_left_bottom_title_text + f" \n(Batch size: {BATCH_SIZE})\n(Samples: {TRAIN_DATA_SAMPLES})",
        color='black',
        pos=(canvas.size[0]//2 - 200, canvas.size[1] - 90),
        anchor_x='center',
        anchor_y='center',
        parent=canvas.scene,
        font_size=10
    )

    fixed_top_title = scene.visuals.Text(
        text=f"Iteration: 0/{TRAIN_DATA_SAMPLES // BATCH_SIZE} | Batch: 0 - {BATCH_SIZE}\nmin f(x,y) = {min_z:.3f}",
        color='black',
        pos=(canvas.size[0]//2, 25),
        anchor_x='center',
        anchor_y='center',
        parent=canvas.scene,
        font_size=8
    )

    # Create a view
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45, distance=50)
    view.camera.azimuth = -19.5
    view.camera.elevation = 18
    view.camera.center = (0, 0, GRAPH_Z_CAMERA_OFFSET)

    scale = (0.5, 0.5, 0.5/10)
    
    # Create colormap for gradient based on Z values
    colormap = Colormap(['blue', 'cyan', 'green', 'yellow', 'red'])
    
    # Normalize Z values to 0-1 range for colormap
    z_min, z_max = Z.min(), Z.max()
    z_normalized = (Z - z_min) / (z_max - z_min)
    
    # Apply colormap to get colors for each vertex
    colors = colormap.map(z_normalized.flatten())
    
    # Create surface plot first with basic color
    surface = scene.visuals.SurfacePlot(
        x=x, 
        y=y, 
        z=Z, 
        color=(0.5, 0.5, 1, 0.6),  # Initial color
        shading='smooth'
    )
    surface.set_gl_state('translucent', depth_test=False)

    surface.transform = STTransform(translate=(0, 0, 0), scale=scale)
    view.add(surface)
    colors[:, 3] = 0.7
    # Now apply the gradient colors to the surface
    # This is the recommended way to set vertex colors after creation
    surface.mesh_data.set_vertex_colors(colors)
    
    # Create sphere with proper scaling
    sphere_radius = 2
    sphere_position = [x_0 * scale[0], y_0 * scale[1], min_fn(x_0, y_0) * scale[2]]
    sphere_scale = (scale[0], scale[1], scale[0])
    sphere = scene.visuals.Sphere(
        radius=sphere_radius,
        color='yellow',
        method='latitude',
        parent=view.scene,
        shading='smooth'
    )
    
    sphere.transform = STTransform(
        translate=sphere_position,
        scale=sphere_scale
    )

    trace_positions = np.array([sphere_position])
    trace = scene.visuals.Line(
        pos=trace_positions,
        color=(1, 1, 1, 0.6),
        width=3,
        parent=view.scene
    )
    
    # Render LaTeX as image
    fi_explain = (
        r"$ \nabla f(\mathbf{x}) ="
        "\begin{bmatrix} \end{bmatrix}"
        "$"
    )

    loss_latex = fr"$ \nabla L(\mathbf{{x}}) = \frac{{1}}{{{BATCH_SIZE}}} \sum_{{i=1}}^{{{BATCH_SIZE}}} \nabla f_i(\mathbf{{x}}) $"

    weight_update = r"$ x_{t+1} \leftarrow x_t - \eta \cdot \nabla L(\mathbf{x_t}) $"


    # if (BATCH_SIZE < TRAIN_DATA_SAMPLES):
    #     fixed_left_bottom_title_text = "Mini-Batch Gradient Descent"
    
    # if (BATCH_SIZE == TRAIN_DATA_SAMPLES):
    #     fixed_left_bottom_title_text = "Gradient Descent"

    if (BATCH_SIZE == 1):
        loss_latex = r"$ \nabla L(\mathbf{x}) = \nabla f_i(\mathbf{x}) $"

    latex = loss_latex + "\n\n" + weight_update

    # latex_img = latex_to_image(latex, dpi=300)

    # # Put formula as overlay (fixed in screen space)
    # image_overlay = scene.visuals.Image(latex_img, parent=canvas.scene)

    # # Position in pixels (center top, for example)
    # image_scale = 0.16
    # image_height = latex_img.shape[0] * image_scale
    # image_width = latex_img.shape[1] * image_scale

    # image_overlay.transform = scene.STTransform(translate=(
    #     view.size[0] - image_width  + 30, 
    #     ( view.size[1] // 2) - image_height//2 + 160, 
    #     0), scale=(image_scale, image_scale, image_scale))


    return view, canvas, sphere, surface, scale, trace, fixed_left_bottom_title, fixed_top_title


view, canvas, sphere, surface, scale, trace, fixed_left_bottom_title, fixed_top_title = init()

# track mouse

def on_mouse_move(event):
    # view.camera.azimuth
    print(view.camera.azimuth, view.camera.elevation, view.camera.distance)

canvas.events.mouse_move.connect(on_mouse_move)


trace_positions = [] 


def update_sphere_position(x, y, z):
    """
    Update sphere position correctly using the same scaling factors
    
    Parameters:
    x, y: World coordinates (before scaling)
    z: World z-coordinate (before scaling)
    """
    # Apply the same scaling used for the surface
    scaled_x = x * scale[0]
    scaled_y = y * scale[1]
    scaled_z = z * scale[2] + 1 # sphere diameter shift
    
    # Get current transform and update only the translation
    current_transform = sphere.transform
    sphere.transform = STTransform(
        translate=[scaled_x, scaled_y, scaled_z],
        scale=current_transform.scale  # Keep the same scale
    )

    trace_positions.append([scaled_x, scaled_y, scaled_z])
    
    # Update trace line
    if len(trace_positions) > 1:
        trace.set_data(pos=np.array(trace_positions))
    
    current_z = z +  SPHERE_D / 2

    global min_z

    if np.abs(current_z) < np.abs(min_z):
        min_z = np.abs(current_z)

    if (BATCH_SIZE < TRAIN_DATA_SAMPLES):
        fixed_left_bottom_title_text = "Mini-Batch\n Gradient\n Descent"
    
    if (BATCH_SIZE == TRAIN_DATA_SAMPLES):
        fixed_left_bottom_title_text = "Gradient\n Descent"

    if (BATCH_SIZE == 1):
        fixed_left_bottom_title_text = "Stochastic\n Gradient\n Descent"



    fixed_left_bottom_title.text = f"{fixed_left_bottom_title_text} \n(Batch size: {BATCH_SIZE})\n(Samples: {TRAIN_DATA_SAMPLES})"

    # Force update
    sphere.update()
    fixed_left_bottom_title.update()


def update_surface():
    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    y = np.linspace(Y_MIN, Y_MAX, SAMPLES)
    X, Y = np.meshgrid(x, y)
    z = min_fn(X, Y)

    surface.set_data(x=x, y=y, z=z)

def df(f, a):
    h = 0.000001
    return ( f(a + h) - f(a) ) / h


def df_x(f, x, y):
    return df(
        lambda x_upd: f(x_upd, y),
        x
    )

def df_y(f, x, y):
    return df(
        lambda y_upd: f(x, y_upd),
        y
    )


def gradient_descent(lp=0.03):
    global x, y

    vx = df_x(min_fn, x, y)
    vy = df_y(min_fn, x, y)

    x = x - vx * lp
    y = y - vy * lp


def frame(_=None):
    global x, y, rand_index, current_epoch

    if current_epoch >= MAX_EPOCHS:
        return

    if (BATCH_SIZE < TRAIN_DATA_SAMPLES):
        rand_index = np.random.randint(0, TRAIN_DATA_SAMPLES - BATCH_SIZE) 

    update_surface()

    gradient_descent(LEARNING_RATE)
    update_sphere_position(x, y, min_fn(x, y))


    current_epoch += BATCH_SIZE / TRAIN_DATA_SAMPLES

    passed_batches = np.round((np.round(current_epoch, 2) % 1) * TRAIN_DATA_SAMPLES, 0).astype(int)
    current_iter = passed_batches // BATCH_SIZE
    max_iter = TRAIN_DATA_SAMPLES // BATCH_SIZE

    # Epoch: {np.round(current_epoch, 2).astype(int)}/{MAX_EPOCHS} | 
    fixed_top_title.text = f"Iteration: {current_iter}/{max_iter} | Batch: {passed_batches } - { passed_batches + BATCH_SIZE }\nmin f(x,y) = {min_z:.3f}"
    fixed_top_title.update()




# timer = app.Timer()
# timer.connect(frame)
# timer.start(1/16)

# clear frames folder
frames_dir = './frames-sgd'
if os.path.exists(frames_dir):
    shutil.rmtree(frames_dir)
os.makedirs(frames_dir)

total_frames = int(TRAIN_DATA_SAMPLES / BATCH_SIZE * MAX_EPOCHS)

for i in range(total_frames):
    # Update your visualization
    # ...
    
    # Render and save frame
    image = canvas.render()
    frame()
    plt.imsave(f'{frames_dir}/frame_{i:03d}.png', image)
    
    # Process events to update display
    canvas.app.process_events()


if __name__ == '__main__':
    app.run()
