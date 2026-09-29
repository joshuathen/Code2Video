from manim import *
import os

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
        self.setup_layout("Conclusion and Real-World Application", [
            "Refraction is collective electron response.",
            "Dispersion stems from varying resonance strengths.",
            "Optical designs must account for this."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        electron_cloud = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.3)
        self.place_at_grid(electron_cloud, 'B3', scale_factor=0.9)
        self.play(Create(electron_cloud))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        spectrum = VGroup(*[Line(UP*0.5, DOWN*0.5, color=c) for c in [RED, GREEN, BLUE]]).arrange(RIGHT, buff=0.1)
        self.place_at_grid(spectrum, 'E2', scale_factor=0.8)
        self.play(Create(spectrum))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Using asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg
        lens_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg"
        if os.path.exists(lens_path):
            lens = SVGMobject(lens_path)
            lens.set_color("#00FFFF")
        else:
            lens = Ellipse(width=1.5, height=0.5, color="#00FFFF")
        self.place_at_grid(lens, 'B6', scale_factor=0.7)
        self.play(FadeIn(lens))
        self.wait(1)
