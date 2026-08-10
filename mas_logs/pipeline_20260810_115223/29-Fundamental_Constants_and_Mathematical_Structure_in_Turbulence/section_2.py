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
        self.setup_layout("Kolmogorov’s Hypothesis: The Universal Constant", 
                          ["K41 theory reveals a universal constant.", 
                           "The inertial subrange follows a power-law.", 
                           "Energy spectrum slope is negative five-thirds."])
        
        # === Animation for Lecture Line 1 ===
        # K41 theory reveals a universal constant.
        self.lecture[0].set_color("#FFD700")
        formula = MathTex(r"E(k) = C \cdot \epsilon^{2/3} \cdot k^{-5/3}", color="#FF5733")
        self.place_in_area(formula, 'A2', 'B5', scale_factor=0.9)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        # The inertial subrange follows a power-law.
        self.lecture[1].set_color("#FFD700")
        arrow = Arrow(start=UP, end=DOWN, color="#2ECC71")
        self.place_at_grid(arrow, 'C2', scale_factor=0.6)
        self.play(Create(arrow))
        
        # === Animation for Lecture Line 3 ===
        # Energy spectrum slope is negative five-thirds.
        self.lecture[2].set_color("#FFD700")
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        plot = axes.plot(lambda x: -1.66 * x, color="#3498DB")
        spectrum_group = VGroup(axes, plot)
        # Placeholder asset loading, handled as a generic icon placeholder as per storyboard
        # icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_in_area(spectrum_group, 'D3', 'F6', scale_factor=0.4)
        self.play(Create(spectrum_group))
        self.wait(2)
