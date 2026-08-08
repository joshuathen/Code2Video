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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup data
        title = "Application: Building the Square Wave"
        lines = [
            "Let's build a square wave using our sine series.",
            "Adding the first harmonic creates a basic smooth curve.",
            "Higher odd harmonics sharpen the corners of the wave.",
            "With enough terms, a jagged square shape emerges.",
            "Watch the \"ears\" appear at the edges—the Gibbs Phenomenon."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_H1 = "#ADD8E6"  # Light Blue
        COLOR_H3 = "#90EE90"  # Light Green
        COLOR_H5 = "#FFFF00"  # Yellow
        COLOR_SUM = WHITE
        COLOR_GIBBS = RED

        # Axes setup
        # Belief B021: Maintain horizontal gap by starting axes at Column 3
        axes = Axes(
            x_range=[-1.1 * PI, 1.1 * PI, PI / 2],
            y_range=[-1.5, 1.5, 1],
            x_length=4.0,
            y_length=3.5,
            axis_config={"include_tip": True, "color": GRAY},
            tips=False
        )
        self.place_in_area(axes, "B3", "F6")

        def fourier_sum_n(x, n_max):
            return sum([(1/k) * np.sin(k*x) for k in range(1, n_max + 1, 2)])

        # === Animation for Lecture Line 1 ===
        # "Let's build a square wave using our sine series."
        self.lecture[0].set_color(COLOR_H1)
        self.play(Create(axes), run_time=1)
        
        func_h1 = lambda x: fourier_sum_n(x, 1)
        graph_current = axes.plot(func_h1, color=COLOR_H1)
        label_current = MathTex("S_1(x) = \\sin(x)", color=COLOR_H1)
        self.place_in_area(label_current, 'A3', 'A5', scale_factor=0.8) # Issue 43 Fix
        
        self.play(Create(graph_current), Write(label_current))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # "Adding the first harmonic creates a basic smooth curve."
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(COLOR_H3)
        
        func_h3 = lambda x: fourier_sum_n(x, 3)
        graph_h3 = axes.plot(func_h3, color=COLOR_H3)
        label_h3 = MathTex("S_3(x) = \\sin(x) + \\frac{1}{3}\\sin(3x)", color=COLOR_H3)
        self.place_in_area(label_h3, 'A3', 'A5', scale_factor=0.8) # Issue 43 Fix
        
        self.play(
            Transform(graph_current, graph_h3),
            Transform(label_current, label_h3),
            run_time=2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "Higher odd harmonics sharpen the corners of the wave."
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(COLOR_H5)
        
        func_h5 = lambda x: fourier_sum_n(x, 5)
        graph_h5 = axes.plot(func_h5, color=COLOR_H5)
        label_h5 = MathTex("S_5(x) = \\sum_{n=1,3,5} \\frac{1}{n}\\sin(nx)", color=COLOR_H5)
        self.place_in_area(label_h5, 'A3', 'A5', scale_factor=0.8) # Issue 43 Fix
        
        self.play(
            Transform(graph_current, graph_h5),
            Transform(label_current, label_h5),
            run_time=2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # "With enough terms, a jagged square shape emerges."
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color(COLOR_SUM)
        
        func_h19 = lambda x: fourier_sum_n(x, 19)
        graph_h19 = axes.plot(func_h19, color=COLOR_SUM)
        label_h19 = MathTex("S_{19}(x) = \\sum_{n=1,3,\\dots,19} \\frac{1}{n}\\sin(nx)", color=COLOR_SUM)
        self.place_in_area(label_h19, 'A3', 'A5', scale_factor=0.8) # Issue 43 Fix
        
        self.play(
            Transform(graph_current, graph_h19),
            Transform(label_current, label_h19),
            run_time=2
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # "Watch the \"ears\" appear at the edges—the Gibbs Phenomenon."
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color(COLOR_GIBBS)
        
        # Asset integration (Issue 29)
        ears_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ears.svg", color=COLOR_GIBBS)
        self.place_at_grid(ears_asset, "E6", scale_factor=0.5)
        
        # Highlighting the overshoot "ears"
        ripples = VGroup(
            Circle(radius=0.15, color=COLOR_GIBBS).move_to(axes.c2p(0.1, 1.2)),
            Circle(radius=0.15, color=COLOR_GIBBS).move_to(axes.c2p(-0.1, -1.2)),
            Circle(radius=0.15, color=COLOR_GIBBS).move_to(axes.c2p(PI-0.1, 1.2)),
            Circle(radius=0.15, color=COLOR_GIBBS).move_to(axes.c2p(-PI+0.1, -1.2))
        )
        
        gibbs_text = Text("Gibbs Phenomenon", color=COLOR_GIBBS)
        # Issue 42 Fix
        self.place_in_area(gibbs_text, 'F3', 'F5', scale_factor=0.6)
        
        self.play(
            Create(ripples),
            FadeIn(ears_asset),
            Write(gibbs_text)
        )
        self.wait(4)
