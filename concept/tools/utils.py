import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def unpack(d: dict, *args):
    return [d[arg] for arg in args]


def unpack_shape(shape):
    res = []
    for d in shape:
        if isinstance(d, (list, tuple)):
            res.extend(unpack_shape(d))
        else:
            res.append(d)
    return res


def assertshape(x, shape, msg=""):
    shape = unpack_shape(shape)
    assert len(x.shape) == len(shape), f"expect {shape=}, got {x.shape=}"
    for i, (x_dim, dim) in enumerate(zip(x.shape, shape)):
        if dim is not None:
            assert x_dim == dim, f"expect dim {i} to be {dim}, got {x_dim}, msg: {msg}"


def animate(clip, filename='animation.mp4', _return=True, fps=10, embed=False):
    # embed = True for Pycharm, otherwise False
    if isinstance(clip, dict):
        clip = clip['image']
    print(f'animating {filename}')
    if filename.endswith('.gif'):
        import imageio
        import matplotlib.image as mpimg
        imageio.mimsave(filename, clip)
        if _return:
            from IPython.display import display
            import ipywidgets as widgets
            return display(widgets.HTML(f'<img src="{filename}" width="750" align="center">'))
        else:
            return

    # from moviepy.editor import ImageSequenceClip
    # clip = ImageSequenceClip(clip, fps=fps)
    # ftype = filename[-3:]
    # if ftype == "mp4":
    #     clip.write_videofile(filename, fps=fps)
    # elif ftype == "gif":
    #     clip.write_gif(filename, fps=fps)
    # else:
    #     raise NotImplementedError(f"file type {ftype} not supported!")
    
    import imageio
    imageio.mimwrite(filename, clip, fps=fps, macro_block_size=1)

    if _return:
        from IPython.display import Video
        return Video(filename, embed=embed, html_attributes="controls autoplay muted loop")


def plt_save_fig_array(close=True, clear=True):
    fig = plt.gcf()
    fig.canvas.draw()
    res = np.array(fig.canvas.renderer.buffer_rgba())
    if close: plt.close()
    if clear: plt.clf()
    return res


def show3view(xyz, ax_front, ax_side, ax_top, lim=2., **scatter_kwargs):
    ax_front.set_title('Front')
    ax_front.scatter(xyz[:, 0], xyz[:, 2], **scatter_kwargs)
    ax_front.set_xlabel('x'); ax_front.set_ylabel('z')
    if lim is not None:
        ax_front.set_aspect('equal'); ax_front.set_xlim(-lim, lim); ax_front.set_ylim(-lim, lim); ax_front.grid()
    ax_side.set_title('Side')
    ax_side.scatter(xyz[:, 1], xyz[:, 2], **scatter_kwargs)
    ax_side.set_xlabel('y'); ax_side.set_ylabel('z')
    if lim is not None:
        ax_side.set_aspect('equal'); ax_side.set_xlim(-lim, lim); ax_side.set_ylim(-lim, lim); ax_side.grid()
    ax_top.set_title('Top')
    ax_top.scatter(xyz[:, 0], xyz[:, 1], **scatter_kwargs)
    ax_top.set_xlabel('x'); ax_top.set_ylabel('y')
    if lim is not None:
        ax_top.set_aspect('equal'); ax_top.set_xlim(-lim, lim); ax_top.set_ylim(-lim, lim); ax_top.grid()


def colorbar(aximshow, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(aximshow, cax=cax)


def flatten_dict(d):
    out = dict()
    for k in d:
        if isinstance(d[k], dict):
            flatten_v = flatten_dict(d[k])
            for k2 in flatten_v:
                out[(k, k2)] = flatten_v[k2]
        else:
            out[k] = d[k]
    return out


def unflatten_dict(d):
    out = dict()
    for k in d:
        if isinstance(k, tuple):
            if k[0] not in out:
                out[k[0]] = dict()
            out[k[0]][k[1]] = d[k]
        else:
            out[k] = d[k]
    return out


def pretty(d, f=None, indent=0):
    if f is None:
        f = lambda v: f"{type(v)}"
    for key, value in d.items():
        print('    ' * indent + str(key), end=': ')
        if isinstance(value, dict):
            print()
            pretty(value, f, indent+1)
        else:
            print(f(value)) # '    ' * (indent+1) + {type(value)} 
            pass

# def countour_plot(f, xrange, yrange, nsamples=100, flevels=50, cmap='Greens'):
#     xlist = np.linspace(*xrange, nsamples)
#     ylist = np.linspace(*yrange, nsamples)
#     X, Y  = np.meshgrid(xlist, ylist)
#     # Z = np.sqrt(X**2 + Y**2)
#     XY = np.stack([X.reshape(-1), Y.reshape(-1)]).T
#     Z = np.array([f(XY[i]) for i in range(XY.shape[0])]).reshape(X.shape)
#     zmin, zmax = np.min(Z), np.max(Z)
#     # fig,ax=plt.subplots(1,1)
#     cp = plt.gca().contourf(X, Y, Z, levels=np.linspace(zmin, zmax, flevels), cmap=cmap)
#     # plt.gcf().colorbar(cp) # Add a colorbar to a plot
#     return X, Y, Z
