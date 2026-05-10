from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def visualize(agent, planned, actual, obstacles, reroute_steps=None, animate=False):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(agent.grid, cmap="Greys", origin="lower")

    # static elements — same as before
    pr, pc = zip(*planned)
    ax.plot(pc, pr, "-", linewidth=2, label="A* initial path", color="tab:blue")

    ax.scatter([planned[0][1]], [planned[0][0]], s=140, marker="o", color="green", label="Start", zorder=5)
    ax.scatter([planned[-1][1]], [planned[-1][0]], s=140, marker="*", color="gold", label="Goal", zorder=5)
    ax.legend(loc="lower right")
    ax.set_title("Drone navigation: A* (blue) + D* Lite reroutes (orange)")
    ax.set_xlabel("col"); ax.set_ylabel("row")

    if not animate:
        # original static plot
        ar, ac = zip(*actual)
        ax.plot(ac, ar, "--", linewidth=2, label="Actual flight", color="tab:orange")
        if obstacles:
            obs_r, obs_c = zip(*obstacles)
            ax.scatter(obs_c, obs_r, s=120, marker="X", color="red", label="Obstacles", zorder=5)
        out = "../flight_plot.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"Plot saved to {out}")
        return

    # animation elements
    drone_dot,  = ax.plot([], [], "o", color="cyan", markersize=10, zorder=6, label="Drone")
    actual_line, = ax.plot([], [], "--", color="tab:orange", linewidth=2, label="Actual flight")
    obs_scatter = ax.scatter([], [], s=120, marker="X", color="red", zorder=5, label="Obstacles")

    flown_r, flown_c = [], []
    shown_obs_r, shown_obs_c = [], []

    reroute_steps = reroute_steps or []

    def update(frame):
        if frame >= len(actual):
            return drone_dot, actual_line, obs_scatter

        r, c = actual[frame]

        # move drone
        drone_dot.set_data([c], [r])

        # draw flown path
        flown_r.append(r)
        flown_c.append(c)
        actual_line.set_data(flown_c, flown_r)

        # show obstacle when reroute happens
        if frame in reroute_steps and frame < len(obstacles):
            obs_r, obs_c = obstacles[frame - reroute_steps.index(frame)]
            shown_obs_r.append(obs_r)
            shown_obs_c.append(obs_c)
            obs_scatter.set_offsets(list(zip(shown_obs_c, shown_obs_r)))

        ax.set_title(f"Drone navigation — step {frame}/{len(actual)}")
        return drone_dot, actual_line, obs_scatter

    anim = FuncAnimation(fig, update, frames=len(actual),
                         interval=80, blit=False)

    out = "../drone_animation.gif"
    anim.save(out, writer="pillow", fps=12)
    print(f"Animation saved to {out}")