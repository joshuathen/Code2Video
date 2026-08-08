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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application: Animal Population", [
            "Watch the rabbit population grow.",
            "Predict numbers using logs.",
            "Model growth with exponents."
        ])
        
        # Define axes
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 10, 2], axis_config={"include_numbers": False}).scale(0.5)
        curve = axes.plot(lambda t: np.exp(0.5 * t), color="#FF9900")
        label = MathTex(r"N(t) = N_0 e^{kt}", font_size=30).set_color("#FF9900")
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(curve, 'D1', 'F6', scale_factor=0.9)
        self.lecture[0].set_color("#FF9900")
        self.play(Create(curve))
        
        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg]
        rabbit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg")
        data_dots = VGroup(*[rabbit.copy().scale(0.1).move_to(axes.c2p(t, np.exp(0.5 * t))) for t in range(5)])
        
        self.place_in_area(data_dots, 'D2', 'E5', scale_factor=0.6)
        self.place_at_grid(label, 'F3', scale_factor=0.7)
        
        self.lecture[1].set_color(BLUE)
        self.play(FadeIn(data_dots), Write(label))
        
        # === Animation for Lecture Line 3 ===
        # Highlight the growth constant k with a pulsing circle.
        k_val = label[0][5] # Extract 'k' (index might vary based on Tex structure, but 'k' is there)
        k_val.set_color(YELLOW)
        circle = Circle(radius=0.2, color=YELLOW).surround(k_val)
        
        self.lecture[2].set_color(YELLOW)
        self.play(Create(circle))
        self.play(Indicate(k_val), run_time=2)
        self.wait(2)
