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
        # Setup layout with titles and 5 lecture lines
        lecture_lines = [
            "Now we extend s to the complex coordinate plane.",
            "We define s with both real and imaginary parts.",
            "The imaginary component creates ripples.",
            "Higher vertical movement increases the frequency.",
            "This maps the entire landscape of the Zeta function."
        ]
        self.setup_layout("Entering the Complex Plane", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=0.5)
        
        # Create coordinate system
        plane_box = Rectangle(width=5.0, height=5.0, color=WHITE, stroke_width=2)
        self.place_in_area(plane_box, "A1", "F6")
        
        re_label = Text("Re", font_size=24, color=WHITE)
        im_label = Text("Im", font_size=24, color=WHITE)
        self.place_at_grid(re_label, "F6", scale_factor=0.6)
        self.place_at_grid(im_label, "A1", scale_factor=0.6)
        
        # Internal static grid lines
        grid_lines = VGroup()
        for i in range(1, 7):
            # Vertical
            grid_lines.add(Line(self.grid[f"A{i}"], self.grid[f"F{i}"], stroke_opacity=0.2))
            # Horizontal
            grid_lines.add(Line(self.grid[f"{chr(65+i-1)}1"], self.grid[f"{chr(65+i-1)}6"], stroke_opacity=0.2))
        
        self.play(Create(plane_box), Create(grid_lines), Write(re_label), Write(im_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        s_point = Dot(color=BLUE)
        self.place_at_grid(s_point, "D3")
        s_formula = Text("s = σ + it", font_size=32, color=BLUE)
        s_formula.next_to(s_point, UR, buff=0.1)
        self.play(FadeIn(s_point), Write(s_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        wave = FunctionGraph(lambda x: 0.4 * np.sin(2 * PI * x), x_range=[0, 4], color=BLUE_B)
        wave.move_to(self.grid["E3"] + RIGHT * 0.5)
        self.play(Create(wave))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        # Move point up (increasing imaginary part t)
        self.play(
            s_point.animate.move_to(self.grid["B3"]),
            s_formula.animate.next_to(self.grid["B3"], UR, buff=0.1),
            run_time=1.5
        )
        # Increase frequency of the wave
        high_freq_wave = FunctionGraph(lambda x: 0.4 * np.sin(5 * PI * x), x_range=[0, 4], color=BLUE_B)
        high_freq_wave.move_to(self.grid["E3"] + RIGHT * 0.5)
        self.play(ReplacementTransform(wave, high_freq_wave))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        self.play(Indicate(plane_box), Indicate(high_freq_wave))
        self.wait(2)
