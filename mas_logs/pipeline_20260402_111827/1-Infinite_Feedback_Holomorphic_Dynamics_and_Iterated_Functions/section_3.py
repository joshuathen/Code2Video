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
        # Data
        title_text = "Escape or Stay: The Fate of Orbits"
        lecture_lines = [
            'The quadratic map shifts points through feedback.',
            'Stable orbits spiral toward a fixed destination.',
            'Unstable orbits accelerate away toward infinity.'
        ]
        
        # Setup layout
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # "The quadratic map shifts points through feedback."
        self.lecture[0].set_color(WHITE)
        formula = Text("f(z) = z^2 + c", color="#FFFFFF")
        # Fix for Issue 32, 33, 34: Use place_in_area, adjust alignment, reduce scale
        self.place_in_area(formula, 'A2', 'A5', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Stable orbits spiral toward a fixed destination."
        self.lecture[1].set_color("#00FF00")
        
        # Complex Plane for visualization
        plane = ComplexPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=3.5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.3},
            axis_config={"stroke_width": 1}
        )
        self.place_in_area(plane, "C2", "F5")
        
        # Stable orbit simulation (using c=0 for clear visual spiraling)
        # z_{n+1} = z_n^2. If |z| < 1, it spirals toward the origin.
        z_s = complex(0.85, 0.4)
        stable_points = [z_s]
        curr_s = z_s
        for _ in range(6):
            curr_s = curr_s**2
            stable_points.append(curr_s)
        
        dots_s = VGroup(*[Dot(plane.n2p(p), radius=0.04, color="#00FF00") for p in stable_points])
        lines_s = VGroup(*[Line(plane.n2p(stable_points[i]), plane.n2p(stable_points[i+1]), stroke_width=2, color="#00FF00") 
                          for i in range(len(stable_points)-1)])
        
        self.play(Create(plane))
        self.play(Create(dots_s), Create(lines_s), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Unstable orbits accelerate away toward infinity."
        self.lecture[2].set_color("#FF0000")
        
        # Unstable orbit starting point (r > 1)
        z_u = complex(1.05, 0.1)
        unstable_points = [z_u]
        curr_u = z_u
        for _ in range(4):
            curr_u = curr_u**2
            unstable_points.append(curr_u)
            if abs(curr_u) > 10: break
            
        dots_u = VGroup(*[Dot(plane.n2p(p), radius=0.04, color="#FF0000") for p in unstable_points])
        lines_u = VGroup(*[Line(plane.n2p(unstable_points[i]), plane.n2p(unstable_points[i+1]), stroke_width=2, color="#FF0000") 
                          for i in range(len(unstable_points)-1)])
        
        self.play(Create(dots_u), Create(lines_u), run_time=2)
        self.wait(2)
