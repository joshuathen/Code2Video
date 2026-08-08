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
        lecture_lines = [
            "In the inertial subrange, universal laws emerge.",
            "Energy density follows a precise mathematical pattern.",
            "Kolmogorov’s K41 theory describes this hidden order.",
            "The energy spectrum decays at a -5/3 rate.",
            "This power law reveals turbulence's mathematical heart."
        ]
        self.setup_layout("The Universal Law: Kolmogorov’s -5/3 Power Scale", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Create log-log axes in white (#FFFFFF) labeled E(k) and k in the visual area.
        self.lecture[0].set_color(YELLOW)
        
        axes = Axes(
            x_range=[0, 2.5, 1],
            y_range=[-3.5, 0.5, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, 'B2', 'E5')
        
        k_label = MathTex("k", color=WHITE)
        e_label = MathTex("E(k)", color=WHITE)
        
        # Issue 33 fix: Move k_label to E6
        self.place_at_grid(k_label, 'E6', scale_factor=0.8)
        self.place_at_grid(e_label, 'B1', scale_factor=0.8)
        
        self.play(Create(axes), Write(k_label), Write(e_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate a jagged white line (#FFFFFF) representing turbulence data across the graph.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        np.random.seed(42)
        def data_func(x):
            # Base slope of -5/3
            base = -5/3 * (x - 0.5) - 0.5
            return base + np.random.normal(0, 0.1)

        points = [axes.c2p(x, data_func(x)) for x in np.linspace(0.2, 2.2, 50)]
        jagged_line = VMobject(color=WHITE)
        jagged_line.set_points_as_corners(points)
        
        self.play(Create(jagged_line), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the linear downward section of the data line in yellow (#FFFF00).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Define the inertial range segment
        x_start, x_end = 0.5, 1.8
        points_highlight = [axes.c2p(x, data_func(x)) for x in np.linspace(x_start, x_end, 30)]
        highlight_line = VMobject(color=YELLOW)
        highlight_line.set_points_as_corners(points_highlight)
        highlight_line.set_stroke(width=6) # Thicker to stand out
        
        self.play(Create(highlight_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Overlay a clean red line (#FF0000) with a slope of -5/3 on the highlighted section.
        # Display label '-5/3' (#FF0000) next to the red slope line.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Clean slope line
        red_line = axes.plot(lambda x: -5/3 * (x - 0.5) - 0.5, x_range=[x_start, x_end], color=RED)
        
        slope_val_label = MathTex("-5/3", color=RED)
        # Issue 34 fix: Move slope_label to D6
        self.place_at_grid(slope_val_label, 'D6', scale_factor=1.0)
        
        self.play(Create(red_line), Write(slope_val_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This power law reveals turbulence's mathematical heart.
        # Flash the label '-5/3' (#FF0000) and display full formula.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        formula = MathTex("E(k) \\approx C \\varepsilon^{2/3} k^{-5/3}", color=WHITE)
        # Issue 35 fix: Place in area A3 to A5
        self.place_in_area(formula, 'A3', 'A5', scale_factor=0.8)
        
        self.play(
            Flash(slope_val_label, color=RED, flash_radius=0.5),
            slope_val_label.animate.scale(1.2).set_color(RED),
            Write(formula)
        )
        self.play(slope_val_label.animate.scale(1/1.2))
        self.wait(2)

        # Finish
        self.lecture[4].set_color(WHITE)
        self.wait(2)
