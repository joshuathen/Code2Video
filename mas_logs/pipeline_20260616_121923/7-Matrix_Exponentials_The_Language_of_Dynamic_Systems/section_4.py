from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lecture_lines = [
            'Eigenvalues determine how particles flow through the system.',
            'Real eigenvalues create straight-line growth or decay.',
            'Imaginary eigenvalues produce oscillating, circular paths.',
            'Predator-prey populations exhibit these cyclical patterns.',
            'The state traces a loop in the phase plane.'
        ]
        self.setup_layout("Visualizing the Flow: Rotation and Scaling", lecture_lines)

        # Value tracker for time-based animation
        time_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        # Step 1: Create a 2D plane with 30 small circular dots (particles) in #FFFFFF at random positions.
        self.lecture[0].set_color(WHITE)
        
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=4, y_length=4,
            axis_config={"include_tip": False, "color": GREY_D}
        )
        # Issue 39: self.place_in_area(axes, 'A1', 'C6', scale_factor=0.8)
        self.place_in_area(axes, "A1", "C6", scale_factor=0.8)
        self.add(axes)

        np.random.seed(42)
        particles = VGroup(*[
            Dot(radius=0.04, color=WHITE).move_to(
                axes.c2p(np.random.uniform(-2.5, 2.5), np.random.uniform(-2.5, 2.5))
            )
            for _ in range(30)
        ])
        # Store initial positions for trajectory calculations
        for p in particles:
            p.initial_pos = axes.p2c(p.get_center())

        self.play(FadeIn(particles))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: For real eigenvalues, animate particles moving along straight lines toward or away from the origin. Color particles #00FF00.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        def update_real_flow(mob):
            t = time_tracker.get_value()
            x0, y0 = mob.initial_pos
            # Simulate decay (x = x0 * e^-0.5t)
            new_x = x0 * np.exp(-0.5 * t)
            new_y = y0 * np.exp(-0.5 * t)
            mob.move_to(axes.c2p(new_x, new_y))

        for p in particles:
            p.add_updater(update_real_flow)
        
        self.play(particles.animate.set_color("#00FF00"))
        self.play(time_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        self.wait(1)

        for p in particles:
            p.remove_updater(update_real_flow)
            p.initial_pos = axes.p2c(p.get_center()) # reset for next step

        # === Animation for Lecture Line 3 ===
        # Step 3: For imaginary eigenvalues, animate particles moving along circular paths around the origin. Color particles #00FFFF.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        time_tracker.set_value(0)
        
        def update_imaginary_flow(mob):
            t = time_tracker.get_value()
            x0, y0 = mob.initial_pos
            r = np.sqrt(x0**2 + y0**2)
            theta = np.arctan2(y0, x0)
            # x = r * cos(theta + 2t), y = r * sin(theta + 2t)
            new_x = r * np.cos(theta + 2 * t)
            new_y = r * np.sin(theta + 2 * t)
            mob.move_to(axes.c2p(new_x, new_y))

        for p in particles:
            p.add_updater(update_imaginary_flow)

        self.play(particles.animate.set_color("#00FFFF"))
        self.play(time_tracker.animate.set_value(PI), run_time=3, rate_func=linear)
        self.wait(1)

        for p in particles:
            p.remove_updater(update_imaginary_flow)
            p.initial_pos = axes.p2c(p.get_center())

        # === Animation for Lecture Line 4 ===
        # Step 4: Display two oscillating wave graphs labeled 'Wolves' [Asset: ...] in #888888 and 'Rabbits' [Asset: ...] in #FFFFFF.
        self.play(self.lecture[3].animate.set_color("#888888"))
        
        graph_axes = Axes(
            x_range=[0, 10, 1], y_range=[-1.5, 1.5, 1],
            x_length=4, y_length=2,
            axis_config={"include_tip": False, "color": GREY_E}
        )
        # Position graph_axes in the bottom rows E and F
        self.place_in_area(graph_axes, "E1", "F6", scale_factor=0.8)
        
        wolf_graph = graph_axes.plot(lambda x: np.sin(x), color="#888888")
        rabbit_graph = graph_axes.plot(lambda x: np.cos(x), color=WHITE)
        
        # Build labels with Assets (Issue 26)
        wolf_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/wolves.svg", color="#888888")
        wolf_text = Text("Wolves", font_size=14, color="#888888")
        wolf_label = VGroup(wolf_icon, wolf_text).arrange(RIGHT, buff=0.1)
        
        rabbit_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/r.svg", color=WHITE)
        rabbit_text = Text("Rabbits", font_size=14, color=WHITE)
        rabbit_label = VGroup(rabbit_icon, rabbit_text).arrange(RIGHT, buff=0.1)

        # Positioning labels (Issue 37, 38)
        self.place_at_grid(wolf_label, "D1", scale_factor=0.8)
        self.place_at_grid(rabbit_label, "D4", scale_factor=0.8)
        
        self.play(
            Create(graph_axes), 
            Create(wolf_graph), 
            Create(rabbit_graph), 
            FadeIn(wolf_label), 
            FadeIn(rabbit_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Step 5: Show a single point tracing a closed loop on the 2D plane, where each coordinate corresponds to the amplitude of the waves from Step 4.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        trace_dot = Dot(color=YELLOW, radius=0.1)
        trace_path = TracedPath(trace_dot.get_center, stroke_color=YELLOW, stroke_width=4)
        
        # Reset tracker for loop (matches sin/cos in Step 4)
        time_tracker.set_value(0)
        trace_dot.move_to(axes.c2p(1, 0)) # x=cos(0)=1, y=sin(0)=0
        
        def update_loop(mob):
            t = time_tracker.get_value()
            # x is Rabbit amplitude (cos), y is Wolf amplitude (sin)
            mob.move_to(axes.c2p(np.cos(t), np.sin(t)))

        self.add(trace_path, trace_dot)
        trace_dot.add_updater(update_loop)
        
        # Dim existing particles to focus on the loop
        self.play(particles.animate.set_opacity(0.2))
        self.play(time_tracker.animate.set_value(TAU), run_time=4, rate_func=linear)
        self.wait(2)
