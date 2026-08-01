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
        # Define content
        title = "The Mystery Meeting"
        lines = [
            "Three math giants meet at a strange crossroads.",
            "Meet e for growth, pi for circles, i for imaginary.",
            "Together, they unlock a beautiful secret of the universe."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors defined in requirements
        color_e = "#FFD700"  # Gold
        color_pi = "#C0C0C0" # Silver
        color_i = "#00FFFF"  # Cyan
        color_silver = "#C0C0C0"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_silver))
        
        # Draw intersecting paths (using Asset SVG)
        crossroads = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/crossroads.svg", color=color_silver)
        self.place_in_area(crossroads, "A1", "F6", scale_factor=2.5)
        
        self.play(Create(crossroads))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The line mentions all three, we use white as base highlight or sequence
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # e symbol and label
        e_math = Text("e", slant=ITALIC, color=color_e)
        e_label = Text("Growth", font_size=24, color=color_e)
        self.place_at_grid(e_math, 'B2', scale_factor=2.5)
        self.place_at_grid(e_label, 'B3', scale_factor=0.8)
        
        # pi symbol and label
        pi_math = Text("π", color=color_pi)
        pi_label = Text("Circles", font_size=24, color=color_pi)
        self.place_at_grid(pi_math, 'D2', scale_factor=2.5)
        self.place_at_grid(pi_label, 'D3', scale_factor=0.8)
        
        # i symbol and label
        i_math = Text("i", slant=ITALIC, color=color_i)
        i_label = Text("Imaginary", font_size=24, color=color_i)
        self.place_at_grid(i_math, 'F2', scale_factor=2.5)
        self.place_at_grid(i_label, 'F3', scale_factor=0.8)
        
        self.play(Write(e_math), FadeIn(e_label))
        self.play(Write(pi_math), FadeIn(pi_label))
        self.play(Write(i_math), FadeIn(i_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Create a central glow
        glow_circle = Circle(radius=0.7, color=WHITE, fill_opacity=0.4, stroke_width=0).move_to(self.grid["D4"])
        glow_dot = Dot(self.grid["D4"], color=WHITE)
        glow = VGroup(glow_circle, glow_dot)
        
        # Merge symbols into the glow
        self.play(
            e_math.animate.move_to(self.grid["D4"]).scale(1.5).set_opacity(0),
            pi_math.animate.move_to(self.grid["D4"]).scale(1.5).set_opacity(0),
            i_math.animate.move_to(self.grid["D4"]).scale(1.5).set_opacity(0),
            FadeOut(e_label), 
            FadeOut(pi_label), 
            FadeOut(i_label),
            FadeOut(crossroads),
            FadeIn(glow)
        )
        self.play(Indicate(glow, scale_factor=1.2))
        self.wait(2)
