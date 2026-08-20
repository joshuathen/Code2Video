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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Photography captures only light intensity.",
            "Holography also records the phase.",
            "Reference and object beams interfere.",
            "This records a 3D wavefront.",
            "The medium locks this relationship."
        ]
        self.setup_layout("Holography: Recording the Wavefront", lecture_lines)
        
        # Mobjects for animations
        laser_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg"
        plate_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg"
        
        ref_beam = SVGMobject(laser_asset, color=WHITE)
        obj_beam = Dot(color=WHITE)
        plate = SVGMobject(plate_asset, color=WHITE)
        
        # Placement fixes (Issue 33)
        self.place_at_grid(plate, 'C6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        # Placement fixes (Issue 34)
        self.place_at_grid(ref_beam, 'B3', scale_factor=0.8)
        self.place_at_grid(obj_beam, 'D3', scale_factor=0.8)
        self.play(Create(ref_beam), Create(obj_beam))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        # Placement fixes (Issue 32)
        interference = VGroup(*[Line(LEFT, RIGHT, color="#FF0000").shift(UP*i*0.2) for i in range(-5, 5)])
        self.place_in_area(interference, 'C3', 'D4', scale_factor=0.4)
        self.play(FadeIn(interference))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        plate.set_color("#00FF00")
        self.play(FadeIn(plate))
        self.wait(2)
