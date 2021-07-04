import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('animation', html='jshtml')
import matplotlib.animation as animation

def test_policy(policy, env, render):

    frames = []

    obs = env.reset()
    done = False

    # number of timesteps so far
    t = 0

    # Logging data
    ep_len = 0            # episodic length
    ep_ret = 0            # episodic return

    while not done:
        t += 1
        if render:
            frames.append(env.render(mode="rgb_array"))

        # Query deterministic action from policy and run it
        action = policy(obs).detach().numpy()
        obs, rew, done, _ = env.step(action)

        # Sum all episodic rewards as we go along
        ep_ret += rew
        
    # Track episodic length
    ep_len = t
    env.close()

    return ep_len, ep_ret, frames

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim