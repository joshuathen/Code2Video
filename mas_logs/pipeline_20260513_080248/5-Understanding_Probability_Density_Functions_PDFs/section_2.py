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

class Section2Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout and Title
        title_text = "Prerequisite: The Smoothing Histogram"
        lecture_lines = [
            'A histogram shows frequencies with rectangular bars.',
            'As we add data and narrow bins, bars shrink.',
            'Jagged steps smooth out into a continuous curve.'
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_ORANGE = "#FFA500"
        COLOR_GREEN = "#00FF00"
        
        # Define Distribution Function (Cheetah Speeds)
        def speed_pdf(x):
            return 0.4 * np.exp(-0.5 * ((x - 5) / 1.5)**2)

        # Helper to generate bars
        def get_hist(axes_obj, num_bins, color):
            dx = 10 / num_bins
            bars = VGroup()
            for i in range(num_bins):
                x_val = i * dx + dx/2
                h_val = speed_pdf(x_val)
                p_bottom = axes_obj.c2p(i * dx, 0)
                p_top = axes_obj.c2p((i+1) * dx, h_val)
                width = p_top[0] - p_bottom[0]
                height = p_top[1] - p_bottom[1]
                bar = Rectangle(
                    width=width,
                    height=height,
                    fill_color=color,
                    fill_opacity=0.7,
                    stroke_color=WHITE,
                    stroke_width=0.5
                )
                bar.move_to(axes_obj.c2p(x_val, h_val/2))
                bars.add(bar)
            return bars

        # === Animation for Lecture Line 1 ===
        # Create axes and place in area (Issue 44)
        histogram_axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 0.5, 0.1],
            axis_config={"include_tip": False, "color": WHITE},
            x_length=5,
            y_length=3
        )
        self.place_in_area(histogram_axes, 'A1', 'D6', scale_factor=0.8)
        
        # Y-axis label (Issue 46)
        y_axis_label = Text("f(x)", font_size=24)
        self.place_at_grid(y_axis_label, 'A2', scale_factor=0.7)
        
        # Jagged histogram
        hist_jagged = get_hist(histogram_axes, 8, COLOR_ORANGE)
        
        # Cheetah icon (Issue 37)
        cheetah_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(cheetah_icon, 'A6', scale_factor=0.6)
        
        self.play(
            self.lecture[0].animate.set_color(COLOR_ORANGE),
            Create(histogram_axes),
            Write(y_axis_label),
            FadeIn(cheetah_icon),
            Create(hist_jagged)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Transition to a finer histogram
        hist_fine = get_hist(histogram_axes, 40, COLOR_ORANGE)
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            Transform(hist_jagged, hist_fine),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Morph into continuous green curve
        smooth_curve = histogram_axes.plot(speed_pdf, color=COLOR_GREEN)
        
        # Probability formula (Issue 45)
        # Using Unicode for mathematical symbols
        probability_formula = Text("P(a ≤ X ≤ b) = ∫ f(x) dx", font_size=24)
        self.place_in_area(probability_formula, 'E1', 'F6', scale_factor=0.9)
        
        # Shaded area to visually link to the formula
        shaded_area = histogram_axes.get_area(smooth_curve, x_range=[3.5, 6.5], color=COLOR_GREEN, opacity=0.3)
        
        self.play(
            self.lecture[2].animate.set_color(COLOR_GREEN),
            ReplacementTransform(hist_jagged, smooth_curve),
            FadeIn(shaded_area),
            Write(probability_formula)
        )
        self.wait(3)

        # Clean Exit
        self.play(
            FadeOut(histogram_axes),
            FadeOut(y_axis_label),
            FadeOut(cheetah_icon),
            FadeOut(smooth_curve),
            FadeOut(shaded_area),
            FadeOut(probability_formula),
            self.lecture.animate.set_color(WHITE)
        )
        self.wait(1)
