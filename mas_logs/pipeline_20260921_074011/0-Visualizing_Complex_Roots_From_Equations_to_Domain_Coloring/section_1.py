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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Challenge of 2D Equations", [
            "Standard graphs fail for complex functions.",
            "We need four dimensions for mapping inputs.",
            "Visualizing complex roots requires new methods."
        ])
        
        # Elements
        complex_plane = ComplexPlane(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_numbers": False})
        func_text = MathTex("f(z) = w").set_color(BLUE)
        real_line = Line(start=LEFT, end=RIGHT, color=RED)
        imag_line = Line(start=DOWN, end=UP, color=GREEN)
        
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        calculator_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#E0E0E0"))
        self.place_in_area(complex_plane, "A3", "C4", scale_factor=0.35)
        self.place_at_grid(func_text, "A3", scale_factor=0.8)
        self.place_at_grid(computer_icon, "B5", scale_factor=0.5)
        self.play(Create(complex_plane), Write(func_text), FadeIn(computer_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.place_at_grid(real_line, "D2", scale_factor=0.6)
        self.place_at_grid(imag_line, "D4", scale_factor=0.6)
        self.play(Create(real_line), Create(imag_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF6666"))
        dot = Dot(color=PURPLE)
        self.place_at_grid(dot, "D3", scale_factor=0.8)
        self.place_at_grid(calculator_icon, "E3", scale_factor=0.5)
        self.play(FadeIn(dot), FadeIn(calculator_icon))
        self.play(dot.animate.move_to(self.grid["C3"]))
        self.wait(1)
