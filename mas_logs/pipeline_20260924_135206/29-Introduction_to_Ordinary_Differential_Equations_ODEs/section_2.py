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
        lecture_lines = ["An ODE links a function to its derivative.", "We solve for functions, not just numbers.", "Take this exponential growth model, for example."]
        self.setup_layout("What is an ODE?", lecture_lines)
        
        # Assets
        bacteria = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bacteria.svg")
        population = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        
        # === Animation for Lecture Line 1 ===
        # Split screen: Left side label 'Function: y = x^2' (color: '#FFFF00').
        # Right side label 'ODE: dy/dx = 2x' (color: '#00FFFF').
        
        func_label = Text("Function: y = x^2", font_size=24, color="#FFFF00")
        ode_label = Text("ODE: dy/dx = 2x", font_size=24, color="#00FFFF")
        
        self.place_at_grid(func_label, 'B2', scale_factor=0.9)
        self.place_at_grid(ode_label, 'B5', scale_factor=0.9)
        
        self.place_at_grid(bacteria, "A2", scale_factor=0.2)
        
        self.lecture[0].set_color("#FFFF00")
        self.play(Write(func_label), Write(ode_label), FadeIn(bacteria))

        # === Animation for Lecture Line 2 ===
        # Draw a static parabola on the left.
        # Animate a set of tangent arrows along the curve on the right.
        
        axes = Axes(x_length=3, y_length=3, x_range=[-2, 2], y_range=[-0.5, 4], axis_config={"include_tip": False})
        parabola = axes.plot(lambda x: x**2, color=WHITE)
        
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.5)
        self.place_in_area(parabola, 'B3', 'E6', scale_factor=0.5)
        
        self.lecture[1].set_color("#00FFFF")
        self.play(Create(axes), Create(parabola))
        
        # Tangent arrows
        tangents = VGroup()
        for x_val in np.linspace(-1.5, 1.5, 5):
            slope = 2 * x_val
            angle = np.arctan(slope)
            arrow = Arrow(start=ORIGIN, end=RIGHT*0.5, color=RED).rotate(angle)
            point = axes.c2p(x_val, x_val**2)
            arrow.move_to(point)
            tangents.add(arrow)
            
        self.play(LaggedStart(*[GrowArrow(t) for t in tangents], lag_ratio=0.2))

        # === Animation for Lecture Line 3 ===
        # Highlight the relationship between the slope of the parabola and the ODE vector field using a connecting line.
        
        self.lecture[2].set_color("#FF00FF")
        
        self.place_at_grid(population, "E2", scale_factor=0.2)
        
        # Highlight relationship
        connection = Line(func_label.get_bottom(), ode_label.get_bottom(), color=YELLOW)
        self.play(Create(connection), FadeIn(population))
        self.wait(1)
