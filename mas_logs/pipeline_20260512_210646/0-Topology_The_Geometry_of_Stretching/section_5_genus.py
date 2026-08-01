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

class Section5GenusScene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Genus counts the number of holes in a surface.',
            'Spheres have genus zero; rings have genus one.',
            'More complex objects have even higher genus values.'
        ]
        self.setup_layout("Classification by Genus (Holes)", lecture_lines)
        
        # Colors
        COLOR_BIN = "#FFFFFF"
        HIGHLIGHT_COLOR = YELLOW

        # Bins setup
        bin_size_x = 1.8
        bin_size_y = 1.2
        
        bin0 = VGroup(
            Rectangle(height=bin_size_y, width=bin_size_x, color=COLOR_BIN),
            Text("Genus 0", font_size=16, color=COLOR_BIN).shift(DOWN * 0.8)
        )
        bin1 = VGroup(
            Rectangle(height=bin_size_y, width=bin_size_x, color=COLOR_BIN),
            Text("Genus 1", font_size=16, color=COLOR_BIN).shift(DOWN * 0.8)
        )
        bin2 = VGroup(
            Rectangle(height=bin_size_y, width=bin_size_x, color=COLOR_BIN),
            Text("Genus 2+", font_size=16, color=COLOR_BIN).shift(DOWN * 0.8)
        )

        self.place_in_area(bin0, 'E1', 'F2')
        self.place_in_area(bin1, 'E3', 'F4')
        self.place_in_area(bin2, 'E5', 'F6')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(FadeIn(bin0), FadeIn(bin1), FadeIn(bin2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Sphere
        sphere = VGroup(
            Circle(radius=0.4, color=WHITE, fill_opacity=0.2),
            Arc(radius=0.4, start_angle=0, angle=PI, color=WHITE).scale(np.array([1, 0.3, 1])),
            Arc(radius=0.4, start_angle=PI, angle=PI, color=WHITE, stroke_opacity=0.3).scale(np.array([1, 0.3, 1]))
        )
        # Issue 47: Sphere B1, scale 1.5
        self.place_at_grid(sphere, 'B1', scale_factor=1.5)
        
        # Ring Asset - Issue 34
        ring = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ring.svg", color=WHITE)
        # Issue 45: Ring B3, scale 1.0
        self.place_at_grid(ring, 'B3', scale_factor=1.0)
        
        # Straw (custom geometry)
        straw = VGroup(
            Line(UP*0.3, DOWN*0.3).shift(LEFT*0.15),
            Line(UP*0.3, DOWN*0.3).shift(RIGHT*0.15),
            Ellipse(width=0.3, height=0.1).shift(UP*0.3),
            Ellipse(width=0.3, height=0.1).shift(DOWN*0.3)
        ).set_color(WHITE)
        # Issue 46: Straw B4, scale 1.0
        self.place_at_grid(straw, 'B4', scale_factor=1.0)

        self.play(FadeIn(sphere))
        self.play(sphere.animate.move_to(bin0[0].get_center()).scale(0.6))
        
        self.play(FadeIn(ring), FadeIn(straw))
        self.play(
            ring.animate.move_to(bin1[0].get_center() + LEFT*0.3).scale(0.6),
            straw.animate.move_to(bin1[0].get_center() + RIGHT*0.3).scale(0.6)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Button Asset - Issue 34
        button = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/button.svg", color=WHITE)
        # Issue 47: Button B5, scale 1.5
        self.place_at_grid(button, 'B5', scale_factor=1.5)
        
        self.play(FadeIn(button))
        self.play(button.animate.move_to(bin2[0].get_center()).scale(0.6))
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
