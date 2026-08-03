from manim import *
import math

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

class Section3Scene(TeachingScene):
    def n_sphere_vol(self, n):
        # Volume of n-sphere with radius R=1
        return (math.pi ** (n / 2)) / math.gamma(n / 2 + 1)

    def construct(self):
        title_text = "The Shrinking Volume Paradox"
        lines = [
            "Volume initially increases as we add dimensions.",
            "It peaks at the fifth dimension then drops.",
            "High-dimensional volume rapidly approaches zero.",
            "The hypercube contains almost all the available space.",
            "Hyperspheres become surprisingly empty in higher dimensions."
        ]
        self.setup_layout(title_text, lines)

        # Color definitions
        color_1 = BLUE_B
        color_2 = "#FFD700"  # GOLD
        color_3 = RED_B
        color_4 = GREEN_B
        color_5 = PURPLE_B

        # === Animation for Lecture Line 1 ===
        axes = Axes(
            x_range=[0, 21, 5],
            y_range=[0, 6, 1],
            x_length=5,
            y_length=3.5,
            axis_config={"include_tip": True, "font_size": 24},
            x_axis_config={"numbers_to_include": [5, 10, 15, 20]},
            y_axis_config={"numbers_to_include": [2, 4, 6]}
        )
        labels = axes.get_axis_labels(x_label="Dim (n)", y_label="Vol")
        graph_group = VGroup(axes, labels)
        self.place_in_area(graph_group, 'C1', 'F6', scale_factor=0.8)

        curve_1 = axes.plot(
            self.n_sphere_vol,
            x_range=[0.01, 5],
            color=color_1
        )

        self.play(self.lecture[0].animate.set_color(color_1))
        self.play(Create(axes), Write(labels))
        self.play(Create(curve_1), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        curve_2 = axes.plot(
            self.n_sphere_vol,
            x_range=[5, 20],
            color=color_2
        )
        
        peak_pos = axes.c2p(5, self.n_sphere_vol(5))
        star = Star(n=5, outer_radius=0.15, inner_radius=0.07, color=color_2, fill_opacity=1)
        star.move_to(peak_pos)

        self.play(self.lecture[1].animate.set_color(color_2))
        self.play(Create(curve_2), run_time=2)
        self.play(FadeIn(star, scale=0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_3))
        
        dot = Dot(color=color_3).move_to(peak_pos)
        self.add(dot)
        self.play(
            MoveAlongPath(dot, curve_2),
            rate_func=linear,
            run_time=2
        )
        self.play(FadeOut(dot))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Unit hypercube vs unit hypersphere metaphor
        # In 2D: Square side 2 (contains circle radius 1)
        square = Square(side_length=1.4, color=WHITE, stroke_width=2)
        circle = Circle(radius=0.7, color=color_4, fill_opacity=0.4)
        
        box_label = Text("Hypercube", font_size=18, color=WHITE)
        sphere_label = Text("Hypersphere", font_size=18, color=color_4)
        
        metaphor_group = VGroup(square, circle)
        self.place_in_area(metaphor_group, 'A1', 'B6', scale_factor=0.8)
        
        # Position labels nearby
        box_label.next_to(square, LEFT, buff=0.2)
        sphere_label.next_to(square, RIGHT, buff=0.2)

        self.play(self.lecture[3].animate.set_color(color_4))
        self.play(
            Create(square), 
            FadeIn(box_label),
            Create(circle),
            FadeIn(sphere_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Dim tracker from 2 to 20
        dim_tracker = ValueTracker(2)
        
        vol_text = Text("Volume:", font_size=20, color=color_5)
        vol_val = DecimalNumber(
            self.n_sphere_vol(2), 
            num_decimal_places=2, 
            color=color_5,
            font_size=20
        )
        vol_display = VGroup(vol_text, vol_val).arrange(RIGHT, buff=0.2)
        self.place_at_grid(vol_display, 'B6', scale_factor=1.0)

        # Scaling logic for visual representation:
        # Ratio of volumes = V_n(1) / 2^n
        # We scale circle area relative to square area by this ratio.
        # Initial ratio (n=2) = pi/4
        # Scale circle radius by sqrt(current_ratio / initial_ratio)
        def get_ratio(n):
            return self.n_sphere_vol(n) / (2**n)
            
        initial_ratio = get_ratio(2)
        circle.save_state()
        
        def update_circle(m):
            n = dim_tracker.get_value()
            ratio = get_ratio(n)
            sf = math.sqrt(ratio / initial_ratio)
            m.restore()
            m.scale(sf)

        circle.add_updater(update_circle)
        vol_val.add_updater(lambda d: d.set_value(self.n_sphere_vol(dim_tracker.get_value())))

        self.play(self.lecture[4].animate.set_color(color_5))
        self.play(FadeIn(vol_display))
        self.play(
            dim_tracker.animate.set_value(20),
            run_time=4,
            rate_func=slow_into
        )
        self.wait(2)
        
        circle.clear_updaters()
        vol_val.clear_updaters()
