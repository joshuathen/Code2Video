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
        lecture_lines = ["We often need population parameters.", "Surveying everyone is impossible.", "Sample means reveal order in chaos."]
        self.setup_layout("The Core Problem: Why Sample Means?", lecture_lines)
        
        # Colors for highlights
        RED = "#FF3333"
        YELLOW = "#FFFF33"
        WHITE = "#FFFFFF"

        # Asset Paths
        ACORN_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/acorn.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(RED)
        # Using SVG asset for acorn as requested in asset integration instruction
        acorn = SVGMobject(ACORN_ASSET)
        self.place_at_grid(acorn, "B4", scale_factor=0.3)
        circle = Circle(color=RED, radius=0.3).move_to(acorn.get_center())
        
        sample_box = Square(color=WHITE, side_length=2.0)
        # Applying requested fix: self.place_in_area(sample_box, 'A3', 'C3', scale_factor=0.7)
        self.place_in_area(sample_box, "A4", "C6", scale_factor=0.7) 
        
        label = Text("Sample", font_size=16).scale(0.7)
        label.next_to(sample_box, UP)
        
        self.play(FadeIn(acorn), Create(circle), Create(sample_box), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        pop_box = Rectangle(color=WHITE, width=4, height=3)
        # Applying requested fix: self.place_in_area(pop_box, 'D3', 'F6', scale_factor=0.6)
        self.place_in_area(pop_box, "D3", "F6", scale_factor=0.6)
        
        pop_label = Text("Population", font_size=18).scale(0.7).next_to(pop_box, UP)
        self.play(FadeIn(pop_box), Write(pop_label))
        
        # Highlight selection using acorns asset
        selections = VGroup(*[SVGMobject(ACORN_ASSET).scale(0.15) for _ in range(5)])
        for i, dot in enumerate(selections):
            dot.move_to(pop_box.get_center() + np.array([(i-2)*0.4, 0, 0]))
        self.play(FadeIn(selections))
        
        avg_line = Line(selections.get_center(), sample_box.get_center(), color=RED)
        self.play(Create(avg_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        squirrel = Text("🐿️", font_size=40)
        # Applying requested fix: self.place_at_grid(squirrel, 'E4', scale_factor=0.6)
        self.place_at_grid(squirrel, "E4", scale_factor=0.6)
        
        forest_bg = Text("Forest", font_size=24, color=GREEN).move_to(pop_box.get_center())
        self.play(FadeIn(squirrel), FadeOut(pop_box), FadeOut(pop_label), FadeOut(selections), FadeOut(avg_line), Write(forest_bg))
        
        summary = Text("Population Estimation", font_size=20, color=WHITE).scale(0.7)
        self.place_at_grid(summary, "A2")
        self.play(Write(summary))
        self.wait(2)
