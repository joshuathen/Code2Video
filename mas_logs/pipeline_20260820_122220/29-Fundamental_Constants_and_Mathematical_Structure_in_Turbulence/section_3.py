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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Kolmogorov Constant (C_K)", [
            "C_K is a universal turbulence constant.",
            "Turbulence follows scale-invariant statistical laws.",
            "We see this in a -5/3 slope."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Equation: E(k) = C_k * epsilon^(2/3) * k^(-5/3)
        eq = MathTex("E(k) = C_K \\cdot \\epsilon^{2/3} \\cdot k^{-5/3}", color=WHITE)
        self.place_in_area(eq, "B2", "B5", scale_factor=1.0)
        self.play(Write(eq))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Flash the symbol C_k
        # Note: Asset path provided in storyboard was /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Since it is a 'none' icon, we apply the highlight color to the MathTex symbol.
        c_k_part = eq.get_part_by_tex("C_K")
        self.play(c_k_part.animate.set_color("#00FF00"), run_time=1)
        self.play(Indicate(c_k_part), run_time=1.5)
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight -5/3 slope on a simple log-log plot
        axes = Axes(x_range=[0, 3], y_range=[0, 3], x_length=4, y_length=4, axis_config={"include_tip": False})
        plot = axes.plot(lambda x: 2 - 0.5 * x, color=BLUE)
        slope_label = MathTex("-5/3 \\text{ slope}", color=BLUE).scale(0.8)
        
        slope_graph = VGroup(axes, plot)
        self.place_in_area(slope_graph, "D2", "F5", scale_factor=0.9)
        self.place_at_grid(slope_label, "D5", scale_factor=0.7)
        
        self.play(Create(axes), Create(plot))
        self.play(Write(slope_label))
        self.lecture[2].set_color("#0000FF")
        self.wait(2)
