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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the scene layout with title and lines
        lecture_lines = [
            "These dynamics mirror feedback loops found throughout our world.",
            "From Romanesco broccoli to fractal antennas, nature iterates.",
            "Infinite complexity emerges from these simple, repeating rules."
        ]
        self.setup_layout("Natural Feedback and Applications", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display a spiral fractal pattern mimicking Romanesco broccoli [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/broccoli.svg] in #32CD32
        romanesco = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/broccoli.svg")
        romanesco.set_color("#32CD32")
        romanesco.height = 1.2 # Baseline height before grid scaling
        
        # VideoCritic Fix (Issue 39): Relocate and scale to avoid crowding
        self.place_in_area(romanesco, "A2", "B5", scale_factor=0.6)
        
        self.play(
            self.lecture[0].animate.set_color("#32CD32"),
            FadeIn(romanesco),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a fractal antenna geometry (Sierpinski Carpet) in #00CED1
        def get_sierpinski_carpet(order, side):
            if order == 0:
                return Square(side_length=side, fill_opacity=1, stroke_width=0, color="#00CED1")
            
            carpet = VGroup()
            new_side = side / 3
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0:
                        continue
                    sub = get_sierpinski_carpet(order - 1, new_side)
                    sub.move_to(i * new_side * RIGHT + j * new_side * UP)
                    carpet.add(sub)
            return carpet

        antenna = get_sierpinski_carpet(3, 1.5)
        
        # VideoCritic Fix (Issue 40): Relocate and scale to avoid overlap
        self.place_in_area(antenna, "C2", "D5", scale_factor=0.7)

        self.play(
            self.lecture[1].animate.set_color("#00CED1"),
            FadeIn(antenna),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the centered text 'The Infinite Feedback of Nature' in #FFFFFF
        final_text = Text(
            "The Infinite Feedback\nof Nature", 
            font_size=32, 
            color="#FFFFFF"
        )
        
        # VideoCritic Fix (Issue 41): Relocate and scale to prevent layout crowding
        self.place_in_area(final_text, "E1", "F6", scale_factor=0.8)

        self.play(
            self.lecture[2].animate.set_color("#FFFFFF"),
            Write(final_text),
            run_time=2
        )
        self.wait(3)
