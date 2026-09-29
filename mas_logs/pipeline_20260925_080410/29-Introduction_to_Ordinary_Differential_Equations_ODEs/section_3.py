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
        self.setup_layout("The First-Order ODE: Growth and Decay", 
                          ["Rate of change equals growth factor.", 
                           "The form is dy/dx = ky.", 
                           "Population increases exponentially over time."])
        
        # Load assets
        bacteria = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bacteria.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#80FF80")
        
        eq = MathTex(r"dy/dx = k y", color=WHITE)
        self.place_at_grid(eq, 'B2', scale_factor=1.2)
        self.place_at_grid(bacteria, 'B4', scale_factor=0.5)
        self.play(Write(eq), FadeIn(bacteria))
        
        k_highlight = MathTex(r"k", color="#80FF80")
        k_highlight.move_to(eq[0][4])
        self.play(ReplacementTransform(eq[0][4].copy(), k_highlight))
        self.wait(1)
        self.remove(k_highlight)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF8080")
        
        axes = Axes(x_length=4, y_length=3, x_range=[0, 5], y_range=[0, 10], axis_config={"include_tip": True})
        graph = axes.plot(lambda x: np.exp(x/1.5), color="#FF8080")
        
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.8)
        self.play(Create(axes), Create(graph))
        self.wait(2)
