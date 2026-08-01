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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize title and lecture lines
        title = "The Geometry Puzzle: The Hypercube Mystery"
        lines = [
            "A tesseract is the four-dimensional version of a cube.",
            "Its surface consists of eight interconnected three-dimensional cubes.",
            "We can visualize it by unfolding it into 3D.",
            "This results in a cross shape made of cubes.",
            "This 3D structure represents a flattened four-dimensional object."
        ]
        self.setup_layout(title, lines)

        # Colors
        TESSERACT_COLOR = "#ADFF2F"
        CUBE_COLOR = "#00FFFF"
        CROSS_COLOR = "#FFD700"

        # Utility for 2D perspective cube
        def get_cube_2d(color=CUBE_COLOR, size=0.5):
            front = Square(side_length=size, color=color, stroke_width=2)
            back = Square(side_length=size, color=color, stroke_width=1).shift(0.2 * size * RIGHT + 0.2 * size * UP)
            connectors = VGroup(*[
                Line(front.get_corner(c), back.get_corner(c), color=color, stroke_width=1)
                for c in [UL, UR, DL, DR]
            ])
            return VGroup(back, connectors, front)

        # === Animation for Lecture Line 1 ===
        # "A tesseract is the four-dimensional version of a cube."
        self.lecture[0].set_color(TESSERACT_COLOR)
        
        # Tesseract projection: inner and outer cube connected
        outer_cube = get_cube_2d(TESSERACT_COLOR, size=1.5)
        inner_cube = get_cube_2d(TESSERACT_COLOR, size=0.7)
        # Shift inner cube slightly to center it within the outer cube projection
        inner_cube.move_to(outer_cube.get_center())
        
        tesseract_connectors = VGroup(*[
            Line(outer_cube[0].get_corner(c), inner_cube[0].get_corner(c), color=TESSERACT_COLOR, stroke_width=1)
            for c in [UL, UR, DL, DR]
        ] + [
            Line(outer_cube[2].get_corner(c), inner_cube[2].get_corner(c), color=TESSERACT_COLOR, stroke_width=1)
            for c in [UL, UR, DL, DR]
        ])
        
        tesseract = VGroup(outer_cube, inner_cube, tesseract_connectors)
        self.place_in_area(tesseract, "B2", "D5")
        
        tesseract_label = Text("4D Tesseract", font_size=18, color=TESSERACT_COLOR)
        # Fix Issue 29: Position at E3 with scale factor
        self.place_at_grid(tesseract_label, 'E3', scale_factor=0.8)
        
        self.play(Create(tesseract), Write(tesseract_label))
        # Simulated rotation
        self.play(tesseract.animate.scale(1.05), run_time=1.5, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Its surface consists of eight interconnected three-dimensional cubes."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(CUBE_COLOR)
        
        self.play(FadeOut(tesseract), FadeOut(tesseract_label))
        
        cubes = VGroup(*[get_cube_2d(CUBE_COLOR, size=0.4) for _ in range(8)])
        positions = ["B2", "B5", "C3", "C5", "D2", "D5", "E3", "E5"]
        for i, pos in enumerate(positions):
            self.place_at_grid(cubes[i], pos)
            
        cubes_label = Text("8 Cubes", font_size=18, color=CUBE_COLOR)
        # Fix Issue 30: Position at E3 with scale factor
        self.place_at_grid(cubes_label, 'E3', scale_factor=0.8)
        
        self.play(FadeIn(cubes), Write(cubes_label))
        self.wait(1)
        
        # Connect at faces (gathering them)
        self.play(
            *[cubes[i].animate.move_to(self.grid["D4"] + np.array([0.15*(i-3.5), 0, 0])) for i in range(8)],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "We can visualize it by unfolding it into 3D."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CROSS_COLOR)
        
        # Define Dali Cross target positions
        # Column: B4, C4, D4, E4
        # Arms: C3, C5, C4_back, C4_front
        target_positions = [
            self.grid["B4"],
            self.grid["C4"],
            self.grid["D4"],
            self.grid["E4"],
            self.grid["C3"],
            self.grid["C5"],
            self.grid["C4"] + 0.35*LEFT + 0.35*UP,  # Pseudo-3D back arm
            self.grid["C4"] + 0.35*RIGHT + 0.35*DOWN  # Pseudo-3D front arm
        ]
        
        self.play(FadeOut(cubes_label))
        cross_label = Text("Unfolded Hypercube", font_size=18, color=CROSS_COLOR)
        # Fix Issue 31 part 1: place_in_area E2-E4
        self.place_in_area(cross_label, 'E2', 'E4', scale_factor=0.7)
        
        self.play(
            *[cubes[i].animate.move_to(target_positions[i]).set_color(CROSS_COLOR) for i in range(8)],
            Write(cross_label),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "This results in a cross shape made of cubes."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(CROSS_COLOR)
        
        # Highlight the cross shape with a pulse
        self.play(cubes.animate.scale(1.1), run_time=0.5)
        self.play(cubes.animate.scale(1/1.1), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This 3D structure represents a flattened four-dimensional object."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(CROSS_COLOR)
        
        highlight = SurroundingRectangle(cubes, color=CROSS_COLOR, buff=0.4, stroke_width=2)
        shadow_label = Text("3D Shadow", font_size=16, color=CROSS_COLOR)
        # Fix Issue 31 part 2: place_at_grid E5
        self.place_at_grid(shadow_label, 'E5', scale_factor=0.7)
        
        self.play(Create(highlight), Write(shadow_label))
        # Final highlight animation
        self.play(
            cubes.animate.set_stroke(width=4),
            highlight.animate.set_stroke(width=4),
            run_time=1
        )
        self.wait(2)
