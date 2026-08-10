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
        self.setup_layout("Diffraction: The Bending of Light", [
            "Diffraction is the bending of waves around obstacles.",
            "Huygens-Fresnel principle: each point acts as a source.",
            "Slits cause light to spread into an intensity pattern."
        ])
        
        # Asset path
        slit_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/slit.svg"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        slit = SVGMobject(slit_path, color=WHITE)
        self.place_at_grid(slit, 'C4', scale_factor=1.0)
        self.play(Create(slit))
        
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        waves = VGroup(*[Line(LEFT*0.5, RIGHT*0.5, color="#FF00FF").shift(LEFT*2 + i*0.3) for i in range(5)])
        self.add(waves)
        self.play(waves.animate.shift(RIGHT*1.8), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        circles = VGroup(*[Arc(radius=0.2 + i*0.2, start_angle=-PI/2, angle=PI, color="#FFFF00") for i in range(3)])
        circles.move_to(slit.get_center())
        self.play(Create(circles), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        diffraction_pattern = FunctionGraph(lambda x: np.sinc(x*2), x_range=[-1.5, 1.5], color="#00FF00")
        self.place_in_area(diffraction_pattern, 'D3', 'F6', scale_factor=0.6)
        
        diffraction_label = Text("Diffraction Pattern", font_size=20, color=WHITE)
        self.place_at_grid(diffraction_label, 'C6', scale_factor=0.7)
        
        self.play(Create(diffraction_pattern), FadeIn(diffraction_label))
        
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        self.wait(2)
