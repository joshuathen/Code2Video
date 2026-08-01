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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial Setup with updated script
        lecture_lines = [
            'Meet five legends of the mathematical world.', 
            'Unity and void: the numbers one and zero.', 
            'Transcendental stars: the constants e and pi.', 
            'The mysterious imaginary unit, i.', 
            'Together, they form the most beautiful equation.'
        ]
        self.setup_layout("The Reunion of Five Great Constants", lecture_lines)
        
        # Define constants using Text and MarkupText
        zero = Text("0", color="#CCCCCC")
        one = Text("1", color="#FFFFFF")
        e_const = Text("e", color="#00BFFF", slant=ITALIC)
        pi_const = Text("π", color="#FFD700")
        i_const = Text("i", color="#FF69B4", slant=ITALIC)
        
        # === Animation for Lecture Line 1 ===
        # Meet five legends of the mathematical world.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Position constants spread across the grid (Fixing Issues 26, 27)
        self.place_at_grid(zero, 'A2', scale_factor=1.5)
        self.place_at_grid(one, 'E5', scale_factor=1.5)
        self.place_at_grid(e_const, 'B2', scale_factor=1.5) 
        self.place_at_grid(pi_const, 'F5', scale_factor=1.5) 
        self.place_at_grid(i_const, 'D2', scale_factor=1.5) 
        
        constants_group = VGroup(zero, one, e_const, pi_const, i_const)
        self.play(FadeIn(constants_group, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Unity and void: the numbers one and zero.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Group 1 and 0 in the center and scale
        self.play(
            zero.animate.scale(1.2).move_to(self.grid['D4']),
            one.animate.scale(1.2).move_to(self.grid['D3']),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transcendental stars: the constants e and pi.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Prepare glows for e and pi
        glow_e = Dot(radius=0.7, color="#00BFFF", fill_opacity=0.3)
        glow_pi = Dot(radius=0.7, color="#FFD700", fill_opacity=0.3)
        self.place_at_grid(glow_e, 'C3')
        self.place_at_grid(glow_pi, 'C4')

        # Bring e and pi to center with glow
        self.play(
            e_const.animate.move_to(self.grid['C3']),
            pi_const.animate.move_to(self.grid['C4']),
            FadeIn(glow_e),
            FadeIn(glow_pi),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The mysterious imaginary unit, i.
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color(WHITE)
        )
        
        # Bring i to center and change color
        self.play(
            i_const.animate.move_to(self.grid['C5']).set_color("#FF1493"),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Together, they form the most beautiful equation.
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(WHITE)
        )
        
        # Create identity equation and place in expanded area (Fixing Issue 28)
        identity = MarkupText('<i>e</i><sup><i>i</i>π</sup> + 1 = 0', color="#FFFFFF")
        self.place_in_area(identity, 'C1', 'D6', scale_factor=1.8)
        
        self.play(
            ReplacementTransform(constants_group, identity),
            FadeOut(glow_e),
            FadeOut(glow_pi),
            run_time=2
        )
        self.wait(2)
