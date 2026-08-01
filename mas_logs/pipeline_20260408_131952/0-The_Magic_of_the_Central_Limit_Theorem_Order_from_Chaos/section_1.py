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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Individual events are often random and unpredictable.',
            'Consider a forest with tiny sprites and massive giants.',
            'This creates a chaotic and bimodal distribution.'
        ]
        self.setup_layout("The Mystery of Predictability", lecture_lines)

        # Color palette
        COLOR_RANDOM = "#FFFF00"
        COLOR_CLUSTER = "#00BFFF"
        COLOR_DIST = "#FF6666"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_RANDOM)

        # Setup Axes
        axes = Axes(
            x_range=[0, 600, 100],
            y_range=[0, 1, 0.2],
            x_length=6,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        x_label = Text("Height", font_size=20, color=WHITE).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("Frequency", font_size=20, color=WHITE).rotate(90*DEGREES).next_to(axes.y_axis, LEFT, buff=0.2)
        axes_group = VGroup(axes, x_label, y_label)
        
        # Fix Issue 39: Move axes_group further right and scale down
        self.place_in_area(axes_group, "A2", "F6", scale_factor=0.75)

        # Randomly jumping data points (Yellow)
        dots = VGroup(*[Dot(color=COLOR_RANDOM, radius=0.06) for _ in range(20)])
        
        # Internal tracker for jump timing
        self.jump_timer = 0
        def update_dots(group, dt):
            self.jump_timer += dt
            if self.jump_timer > 0.15:
                for dot in group:
                    rx = np.random.uniform(5, 595)
                    ry = np.random.uniform(0.1, 0.8)
                    dot.move_to(axes.c2p(rx, ry))
                self.jump_timer = 0

        self.add(axes_group, dots)
        dots.add_updater(update_dots)
        self.play(FadeIn(axes_group), FadeIn(dots), run_time=1)
        self.wait(3)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_CLUSTER)

        # Remove jitter
        dots.remove_updater(update_dots)
        
        # Move dots to horizontal line (hidden axis) then to clusters
        cluster_animations = []
        for i, dot in enumerate(dots):
            if i < 10: # Sprites
                target_x = np.random.normal(30, 15)
            else: # Giants
                target_x = np.random.normal(500, 30)
            target_y = np.random.uniform(0.05, 0.5)
            cluster_animations.append(dot.animate.set_color(COLOR_CLUSTER).move_to(axes.c2p(target_x, target_y)))

        sprites_label = Text("Sprites", font_size=20, color=WHITE)
        giants_label = Text("Giants", font_size=20, color=WHITE)
        
        # Fix Issue 40 & 41: Precise grid positioning
        self.place_at_grid(sprites_label, 'B2', scale_factor=0.8)
        self.place_at_grid(giants_label, 'B5', scale_factor=0.8)

        self.play(
            *cluster_animations,
            Write(sprites_label),
            Write(giants_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_DIST)

        # Bimodal Distribution Curve
        dist_curve = axes.plot(
            lambda x: 0.65 * (np.exp(-((x - 30) ** 2) / (2 * 25 ** 2)) + np.exp(-((x - 500) ** 2) / (2 * 45 ** 2))),
            color=COLOR_DIST,
            x_range=[0, 600]
        )

        self.play(Create(dist_curve), run_time=2)
        self.wait(3)
