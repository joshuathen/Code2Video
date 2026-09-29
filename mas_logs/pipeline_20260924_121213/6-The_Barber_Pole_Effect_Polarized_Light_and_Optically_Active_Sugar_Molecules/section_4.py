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
        self.setup_layout("Mathematical Visualization: The Rotation Vector", [
            "Rotation depends on concentration and length.",
            "Changing concentration shifts the output vector.",
            "The graph visualizes this precise shift."
        ])
        
        # UI elements
        light_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")
        circle = Circle(radius=1.5, color="#555555")
        
        # Apply positioning constraints from issues
        self.place_at_grid(circle, 'B4', scale_factor=0.9)
        self.place_at_grid(light_icon, 'B4', scale_factor=0.3)
        
        # Vector
        vector = Arrow(start=ORIGIN, end=UP*1.2, color="#00FFFF", buff=0)
        self.place_at_grid(vector, 'B4', scale_factor=0.9)
        
        label_v = Text("v", font_size=24, color="#00FFFF")
        label_v.next_to(vector.get_end(), UP, buff=0.1)
        
        # Equation (B040: central row D)
        equation = MathTex(r"\\theta_{rot} = [\\alpha] \\cdot c \\cdot l", color="#FFFFFF")
        self.place_at_grid(equation, 'D4', scale_factor=0.8)
        
        # Variables trackers
        concentration = ValueTracker(0)
        
        # Updater for rotation
        vector.add_updater(lambda mob: mob.put_start_and_end_on(
            self.grid['B4'],
            self.grid['B4'] + np.array([
                1.2 * np.sin(concentration.get_value()),
                1.2 * np.cos(concentration.get_value()),
                0
            ])
        ))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.play(Create(circle), FadeIn(light_icon), Create(vector), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color("#00FF00")
        self.play(Write(equation))
        self.play(concentration.animate.set_value(-PI/2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
