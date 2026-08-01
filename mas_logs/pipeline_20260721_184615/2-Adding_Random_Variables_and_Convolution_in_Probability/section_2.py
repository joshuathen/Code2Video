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
        title = "Prerequisite Check: Independence & PDFs"
        lines = [
            "Assume these jars act independently of each other.",
            "Each jar's uncertainty is described by a PDF.",
            "Their joint probability is the product of individual PDFs."
        ]
        
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#A9A9A9")
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/jar.svg]
        jar_a_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/jar.svg", color="#A9A9A9")
        jar_b_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/jar.svg", color="#A9A9A9")
        
        label_jar_a = Text("Jar A", font_size=18, color="#FFFFFF")
        label_jar_b = Text("Jar B", font_size=18, color="#FFFFFF")
        
        jar_a = VGroup(jar_a_svg, label_jar_a).arrange(UP, buff=0.2)
        jar_b = VGroup(jar_b_svg, label_jar_b).arrange(UP, buff=0.2)
        
        # Fixed positioning per VideoCritic issues 42 & 43
        self.place_at_grid(jar_a, "B3", scale_factor=0.8)
        self.place_at_grid(jar_b, "B6", scale_factor=0.8)
        
        # Dashed line between jars representing independence/separation
        dashed_line = DashedLine(
            start=self.grid["B4"],
            end=self.grid["B5"],
            color="#A9A9A9"
        )
        
        self.play(Create(jar_a), Create(jar_b), run_time=1.0)
        self.play(Create(dashed_line), run_time=0.5)
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00") 
        
        # Bell curve function: f(x) = exp(-x^2)
        def bell_curve(x):
            return np.exp(-x**2)
        
        # Green PDF for Jar A
        axes_a = Axes(x_range=[-2, 2], y_range=[0, 1.2], x_length=1.5, y_length=1.0, 
                      axis_config={"include_tip": False, "color": "#A9A9A9"}).scale(0.7)
        curve_a = axes_a.plot(bell_curve, color="#00FF00")
        label_a = MathTex("f(x)", font_size=20, color="#00FF00").next_to(axes_a, UP, buff=0.1)
        pdf_a = VGroup(axes_a, curve_a, label_a)
        
        # Blue PDF for Jar B
        axes_b = Axes(x_range=[-2, 2], y_range=[0, 1.2], x_length=1.5, y_length=1.0, 
                      axis_config={"include_tip": False, "color": "#A9A9A9"}).scale(0.7)
        curve_b = axes_b.plot(bell_curve, color="#00BFFF")
        label_b = MathTex("g(y)", font_size=20, color="#00BFFF").next_to(axes_b, UP, buff=0.1)
        pdf_b = VGroup(axes_b, curve_b, label_b)
        
        # Fixed positioning per VideoCritic issues 42 & 43
        self.place_at_grid(pdf_a, "D3", scale_factor=1.0)
        self.place_at_grid(pdf_b, "D6", scale_factor=1.0)
        
        self.play(Create(pdf_a), run_time=1.0)
        self.play(Create(pdf_b), run_time=1.0)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFFFF")
        
        # Joint probability formula
        joint_formula = MathTex(
            "P(X, Y) = f(x) \\cdot g(y)",
            font_size=32,
            color="#FFFFFF"
        )
        
        # Position formula per VideoCritic issue 44
        self.place_in_area(joint_formula, "F3", "F6", scale_factor=1.0)
        
        self.play(Write(joint_formula), run_time=1.2)
        self.play(Indicate(joint_formula, color="#FFFFFF"))
        self.wait(2.0)
