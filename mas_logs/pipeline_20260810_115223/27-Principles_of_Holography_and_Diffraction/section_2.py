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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Diffraction is light bending around obstacles.",
            "Huygens-Fresnel: each point acts as a source.",
            "Apertures create distinct interference patterns."
        ]
        self.setup_layout("The Core Mechanism: Diffraction", lecture_lines)
        
        # Animations
        # Create objects
        wavefront = Line(start=UP*1.0, end=DOWN*1.0, color=WHITE).rotate(PI/2)
        pinhole = Line(start=UP*0.5, end=DOWN*0.5, color=GREY).rotate(PI/2)
        wavelets = VGroup(*[Circle(radius=0.1+i*0.2, color=GREEN, stroke_width=2) for i in range(3)])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_in_area(wavefront, 'B3', 'B5', scale_factor=0.5)
        self.place_at_grid(pinhole, 'C4', scale_factor=0.6)
        self.play(Create(wavefront), Create(pinhole))
        self.play(wavefront.animate.shift(RIGHT*1.0))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.place_at_grid(wavelets, 'D4', scale_factor=0.4)
        self.play(Create(wavelets))
        self.play(wavelets.animate.scale(2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        pattern = VGroup(*[Circle(radius=0.3+i*0.2, color=WHITE, stroke_width=1) for i in range(4)])
        self.place_in_area(pattern, 'C3', 'E5', scale_factor=0.6)
        self.play(FadeIn(pattern))
        self.wait(2)
