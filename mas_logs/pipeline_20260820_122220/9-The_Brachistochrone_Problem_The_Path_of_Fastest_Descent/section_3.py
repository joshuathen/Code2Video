from manim import *

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
        self.setup_layout("The Mathematical Framework: Functional Calculus", 
                         ["We seek a function that minimizes travel time.", 
                          "This requires the calculus of variations.", 
                          "Mathematically, we solve for an optimal shape."])
        
        # === Animation for Lecture Line 1 ===
        # Write the functional integral S = integral f(y, y') dx.
        integral_formula = MathTex(r"S = \int_{x_0}^{x_1} f(y, y') \, dx", font_size=36)
        self.place_in_area(integral_formula, 'A2', 'B5', scale_factor=0.8)
        self.play(Write(integral_formula))
        self.lecture[0].set_color("#FFD700") # Gold

        # === Animation for Lecture Line 2 ===
        # Identify key functional elements in #FF4500.
        highlight = SurroundingRectangle(integral_formula[0][6], color="#FF4500", buff=0.1)
        self.play(Create(highlight))
        self.lecture[1].set_color("#FF4500")

        # === Animation for Lecture Line 3 ===
        # Show variation delta S = 0 as the condition, #1E90FF.
        condition = MathTex(r"\delta S = 0", font_size=40, color="#1E90FF")
        self.place_at_grid(condition, 'C3', scale_factor=0.9)
        self.play(Write(condition))
        self.lecture[2].set_color("#1E90FF")
        
        self.wait(2)
