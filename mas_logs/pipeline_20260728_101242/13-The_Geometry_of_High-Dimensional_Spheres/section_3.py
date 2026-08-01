from manim import *
import numpy as np
from scipy.special import gamma

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
        self.setup_layout(
            "The Volume Paradox: Shrinking to Zero",
            [
                "Unit sphere volume changes as dimensions increase.",
                "Volume peaks at dimension five then starts decreasing.",
                "In very high dimensions, the volume approaches zero.",
                "Compare a sphere inside a high-dimensional cube.",
                "The sphere occupies almost none of the cube's volume."
            ]
        )

        # Colors
        c1 = "#FFFFFF" # White for formula
        c2 = "#00BFFF" # DeepSkyBlue for chart
        c3 = "#00BFFF" # Same for shrinking bars
        c4 = "#FFFFFF" # White for Cube
        c5 = "#00FF00" # Green for Sphere

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        
        # Formula V_n(1) = (pi^(n/2)) / Gamma(n/2 + 1)
        formula = MathTex(
            r"V_n(1) = \frac{\pi^{n/2}}{\Gamma(\frac{n}{2} + 1)}",
            color=c1
        )
        # Fix Issue 29: place in A2-A6, scale 0.8
        self.place_in_area(formula, 'A2', 'A6', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        
        def get_volume(n):
            return (np.pi**(n/2)) / gamma(n/2 + 1)
        
        vol_values = [get_volume(n) for n in range(1, 11)]
        
        # Fix Issue 28: place in B2-F6, scale 0.6
        chart = BarChart(
            values=vol_values,
            bar_names=[str(i) for i in range(1, 11)],
            y_range=[0, 6, 2],
            y_axis_config={"font_size": 18},
            x_axis_config={"font_size": 18},
            bar_colors=[c2] * 10
        )
        self.place_in_area(chart, 'B2', 'F6', scale_factor=0.6)
        
        # Animate growth up to n=5
        self.play(Create(chart.x_axis), Create(chart.y_axis))
        self.play(LaggedStart(*[Create(bar) for bar in chart.bars[:5]], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))
        
        # Animate bars n=6 to 10
        self.play(LaggedStart(*[Create(bar) for bar in chart.bars[5:]], lag_ratio=0.2))
        
        # Volume -> 0 label
        vol_text = Text("Volume → 0", color=c3, font_size=24)
        # Fix Issue 27: place at B5, scale 0.8
        self.place_at_grid(vol_text, 'B5', scale_factor=0.8)
        self.play(Write(vol_text))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(c5))
        
        # Transition from chart to Cube/Sphere comparison
        self.play(FadeOut(chart), FadeOut(vol_text), FadeOut(formula))
        
        # Cube (Square) and Sphere (Circle)
        cube = Square(side_length=3.0, color=c4)
        self.place_in_area(cube, 'B2', 'F6', scale_factor=1.0)
        
        circle = Circle(radius=1.5, color=c5, fill_opacity=0.3)
        circle.move_to(cube.get_center())
        
        cube_label = Text("Cube", color=c4, font_size=20)
        sphere_label = Text("Sphere", color=c5, font_size=20)
        
        # Position labels
        self.place_at_grid(cube_label, 'B2', scale_factor=0.8)
        self.place_at_grid(sphere_label, 'B3', scale_factor=0.8)
        
        self.play(Create(cube), Write(cube_label))
        self.play(Create(circle), Write(sphere_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(c5))
        
        # Scale circle down and fade
        self.play(
            circle.animate.scale(0.1).set_fill(opacity=0.05).set_stroke(opacity=0.1),
            sphere_label.animate.scale(0.8).move_to(self.grid['E4']),
            run_time=2
        )
        self.wait(2)
