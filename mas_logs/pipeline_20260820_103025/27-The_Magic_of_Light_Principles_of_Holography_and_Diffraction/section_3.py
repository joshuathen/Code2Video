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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Holographic Recording Process", [
            "Reference beams meet object reflections.", 
            "Interference patterns capture intensity and phase.", 
            "Recording the encoded 3D information.", 
            "Phases shift across the holographic film.", 
            "Mapping ripples to data structure."
        ])
        
        # Assets
        rabbit_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg"
        laser_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg"
        fringe_path = "/scratch/pawsey1357/jthen/Code2Video/assets/rabbit_fringe_pattern.png"

        rabbit = SVGMobject(rabbit_path) if os.path.exists(rabbit_path) else Dot(color="#FFD700")
        laser = SVGMobject(laser_path) if os.path.exists(laser_path) else Line(LEFT, RIGHT, color="#FF0000")
        plate = Rectangle(width=1.5, height=2.5, color="#888888")
        
        # Layout
        self.place_at_grid(rabbit, "B2", scale_factor=0.7) # Issue 29/38
        self.place_at_grid(laser, "C3", scale_factor=0.7) # Issue 28/38
        self.place_at_grid(plate, "D5", scale_factor=0.6) # Issue 27/38

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.play(FadeIn(rabbit), FadeIn(laser), FadeIn(plate))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#AAAAAA")
        self.play(self.lecture[1].animate.set_opacity(1))
        dots = VGroup(*[Dot(radius=0.03, color=BLUE) for _ in range(20)])
        dots.arrange_in_grid(4, 5)
        self.place_at_grid(dots, "D4", scale_factor=0.5) # Issue 27/38
        self.play(FadeIn(dots))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00DDFF")
        self.play(self.lecture[2].animate.set_opacity(1))
        fringe = ImageMobject(fringe_path) if os.path.exists(fringe_path) else Square(color="#FFFF00")
        self.place_at_grid(fringe, "D6", scale_factor=0.4) # Issue 27/38
        self.play(FadeIn(fringe))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(self.lecture[3].animate.set_opacity(1))
        self.play(fringe.animate.shift(LEFT*0.1), run_time=0.5)
        self.play(fringe.animate.shift(RIGHT*0.1), run_time=0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.play(self.lecture[4].animate.set_opacity(1))
        plate.set_color("#00FF00")
        self.play(Indicate(plate))
