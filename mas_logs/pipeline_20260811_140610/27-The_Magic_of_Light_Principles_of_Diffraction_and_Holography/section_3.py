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
        self.setup_layout("The Core Concept: Interference Patterns as Information", [
            "Holography is recorded interference patterns.", 
            "We use reference and object beams.", 
            "The pattern captures phase and amplitude.", 
            "It stores information about the object.", 
            "The image is essentially frozen light."
        ])
        
        # Load assets
        laser1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color="#00FFFF")
        laser2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color="#00FFFF")
        
        # Group assets in safe grid zone A4-F6 (B004)
        wave_group = VGroup(laser1, laser2)
        self.place_in_area(wave_group, 'A4', 'C6', scale_factor=0.7)

        # Setup interference pattern
        fringes = VGroup(*[Line(UP*0.5, DOWN*0.5, color="#FFFF00", stroke_width=3) for i in range(10)])
        fringes.arrange(RIGHT, buff=0.1)
        self.place_at_grid(fringes, 'D4', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.play(FadeIn(wave_group))
        self.play(laser1.animate.shift(LEFT*0.5), laser2.animate.shift(RIGHT*0.5), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_opacity(1)
        self.play(Create(fringes))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_opacity(1)
        # Shift one source [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg]
        self.play(laser1.animate.shift(UP*0.5).set_color("#FF00FF"), fringes.animate.set_color("#FF00FF"), run_time=1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_opacity(1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_opacity(1)
        self.play(FadeOut(wave_group), fringes.animate.set_opacity(0.5))
        self.wait(2)
