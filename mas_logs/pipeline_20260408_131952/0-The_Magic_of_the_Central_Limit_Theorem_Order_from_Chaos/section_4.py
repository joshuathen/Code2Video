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
        # Initial Setup
        title = "The Emergence of the Bell Curve"
        lines = [
            'Watch the original chaotic distribution begin to fade.',
            'Hundreds of sample means stack into new columns.',
            'An unmistakable bell-shaped outline starts to appear.',
            'This is the smooth, mathematical Normal Distribution.',
            'Witness the perfect transition from chaos to order.'
        ]
        self.setup_layout(title, lines)

        # Pre-create Upper Bimodal Distribution
        def bimodal_func(x):
            return 1.5 * (np.exp(-((x - 1)**2) / 0.4) + np.exp(-((x + 1)**2) / 0.4))

        upper_axes = Axes(
            x_range=[-3, 3], y_range=[0, 2],
            axis_config={"include_tip": False},
            x_length=4.5, y_length=1.5
        ).set_color(WHITE)
        bimodal_curve = upper_axes.plot(bimodal_func, color=WHITE)
        bimodal_group = VGroup(upper_axes, bimodal_curve)
        
        # Issue 49: Place in area A2-B5 with scale 0.9
        self.place_in_area(bimodal_group, "A2", "B5", scale_factor=0.9)
        self.add(bimodal_group)

        # Pre-create Lower Axes
        lower_axes = Axes(
            x_range=[-3, 3], y_range=[0, 10],
            axis_config={"include_tip": False},
            x_length=4.5, y_length=2.5
        ).set_color(WHITE)
        
        # Issue 50: Place in area D1-F6 with scale 0.8
        self.place_in_area(lower_axes, "D1", "F6", scale_factor=0.8)
        self.add(lower_axes)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#333333")
        self.play(
            bimodal_group.animate.set_color("#333333"),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        
        # Setup for stacking balls [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/balls.svg]
        ball_template = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/balls.svg").set_height(0.08).set_color(WHITE)
        
        num_bins = 18
        bin_edges = np.linspace(-2.4, 2.4, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Generate target distribution counts (Normal approximation)
        mu, sigma = 0, 0.75
        total_balls = 200 # Reduced count for performance, looks like 'hundreds'
        probs = np.exp(-0.5 * ((bin_centers - mu) / sigma)**2)
        probs /= probs.sum()
        bin_counts = (probs * total_balls).astype(int)

        all_balls = VGroup()
        ball_step_y = 0.085
        
        for i, count in enumerate(bin_counts):
            x_coord = lower_axes.c2p(bin_centers[i], 0)[0]
            y_base = lower_axes.c2p(0, 0)[1]
            for j in range(count):
                ball = ball_template.copy()
                ball.move_to([x_coord, y_base + (j * ball_step_y) + 0.05, 0])
                all_balls.add(ball)

        # Batch drop balls for impact and speed
        batch_size = 20
        batches = [all_balls[i:i + batch_size] for i in range(0, len(all_balls), batch_size)]
        
        for batch in batches:
            self.play(
                FadeIn(batch, shift=UP*0.3),
                run_time=0.15
            )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        # Outline of the bell shape
        bell_outline = lower_axes.plot(
            lambda x: 8.5 * np.exp(-0.5 * (x / 0.75)**2),
            color="#00FF00",
            x_range=[-2.4, 2.4],
            stroke_width=4
        )
        
        self.play(
            all_balls.animate.set_color("#00FF00"),
            Create(bell_outline),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#1E90FF")
        
        # Mathematical curve
        normal_curve = lower_axes.plot(
            lambda x: 8.5 * np.exp(-0.5 * (x / 0.75)**2),
            color="#1E90FF",
            stroke_width=6
        )
        
        curve_label = Text("Normal Distribution", color="#1E90FF", font_size=24)
        # Issue 48: Place label in C1-C3 at scale 0.8
        self.place_in_area(curve_label, 'C1', 'C3', scale_factor=0.8)

        self.play(
            Create(normal_curve),
            Write(curve_label),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#1E90FF")
        
        # Emphasize the final transition
        self.play(
            Flash(normal_curve, color="#1E90FF", line_length=0.5, num_lines=20),
            normal_curve.animate.set_stroke(width=8),
            run_time=1.5
        )
        self.wait(2)
