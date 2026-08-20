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
        self.setup_layout("Defining Failure: The Cost Function", [
            "The cost function measures prediction error.",
            "We represent this as an error landscape.",
            "Low valleys signify high model accuracy."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display 'Cost Function: Mean Squared Error'
        mse_text = Text("Cost Function: MSE", font_size=24, color=WHITE)
        self.place_at_grid(mse_text, "A4", scale_factor=0.9)
        self.play(Write(mse_text))
        self.lecture[0].set_color("#FFCC00")

        # === Animation for Lecture Line 2 ===
        # Load asset landscape
        error_landscape_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/landscape.svg")
        error_landscape_svg.set_color("#888888")
        
        self.place_in_area(error_landscape_svg, "B3", "E5", scale_factor=0.75)
        
        self.play(Create(error_landscape_svg))
        self.lecture[1].set_color("#888888")

        # Place 'Current Guess'
        dot = Dot(color="#FFCC00")
        dot.move_to(error_landscape_svg.get_top() + RIGHT * 0.5)
        self.add(dot)
        self.play(FadeIn(dot))

        # === Animation for Lecture Line 3 ===
        # Highlight Global Minimum
        min_label = Text("Global Minimum", font_size=18, color="#00FF00")
        self.place_at_grid(min_label, "F4", scale_factor=0.9)
        
        target = Dot(color="#00FF00")
        target.move_to(error_landscape_svg.get_bottom())
        
        self.play(FadeIn(target), Write(min_label))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
