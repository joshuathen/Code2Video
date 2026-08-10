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
        self.setup_layout("Visualizing Gradient Descent", [
            "The loss landscape is a hilly terrain.",
            "The current error is your altitude.",
            "We take steps downhill to minimize error.",
            "This path is known as gradient descent.",
            "A hiker finds the valley of minimum error."
        ])
        
        # Elements
        landscape = FunctionGraph(lambda x: 0.2 * np.sin(3 * x) + 0.1 * x**2, x_range=[-3, 3], color=BLUE)
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg")
        math_tex = MathTex("w_{new} = w_{old} - \\eta \\cdot \\nabla Loss", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#87CEEB")
        self.place_in_area(landscape, 'B2', 'E5', scale_factor=0.4)
        self.play(Create(landscape))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6347")
        self.place_at_grid(hiker, 'C3', scale_factor=0.7)
        self.play(FadeIn(hiker))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        descent_path = Line(start=ORIGIN, end=DOWN*0.5 + RIGHT*0.3, color=YELLOW)
        self.place_in_area(descent_path, 'C3', 'D4', scale_factor=0.6)
        self.play(hiker.animate.move_to(descent_path.get_end()), Create(descent_path))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFA500")
        self.place_in_area(math_tex, 'E2', 'E5', scale_factor=0.7)
        self.play(Write(math_tex))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#32CD32")
        self.play(hiker.animate.move_to(self.grid['E4']))
        self.wait(1)
