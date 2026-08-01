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

class Section4Scene(TeachingScene):
    def construct(self):
        # Data
        title = "Analytic Continuation: The Secret Bridge"
        lines = [
            "Analytic continuation extends the function's reach beyond its boundary.",
            "Think of it as completing a hidden mathematical puzzle.",
            "A flashlight illuminates one side of a dark room.",
            "Continuation turns on a light for the entire space.",
            "The new values match the original patterns perfectly."
        ]
        
        self.setup_layout(title, lines)

        # Puzzle grid setup (right side is the grid A1-F6)
        # We will split it into Right (4,5,6) and Left (1,2,3)
        right_puzzle = VGroup()
        left_puzzle = VGroup()
        
        for r in ["A", "B", "C", "D", "E", "F"]:
            for c in ["1", "2", "3"]:
                piece = RoundedRectangle(corner_radius=0.1, width=0.8, height=0.8, color=WHITE, fill_opacity=0.2)
                self.place_at_grid(piece, f"{r}{c}")
                left_puzzle.add(piece)
            for c in ["4", "5", "6"]:
                piece = RoundedRectangle(corner_radius=0.1, width=0.8, height=0.8, color=WHITE, fill_opacity=0.2)
                self.place_at_grid(piece, f"{r}{c}")
                right_puzzle.add(piece)

        # Boundary line
        boundary = DashedLine(
            start=self.grid["A3"] + RIGHT*0.5 + UP*0.5,
            end=self.grid["F3"] + RIGHT*0.5 + DOWN*0.5,
            color=GRAY
        )
        boundary_label = Text("Boundary", font_size=16, color=GRAY)
        boundary_label.next_to(boundary, UP)

        # Flashlights
        # Right flashlight beam
        right_beam = Polygon(
            self.grid["D6"] + RIGHT*0.5, 
            self.grid["A4"] + LEFT*0.5 + UP*0.5, 
            self.grid["F4"] + LEFT*0.5 + DOWN*0.5,
            color="#FFFF00", fill_opacity=0.3, stroke_width=0
        )
        right_source = Star(n=5, color="#FFFF00", fill_opacity=1)
        self.place_at_grid(right_source, "D6", scale_factor=0.8)

        # Left flashlight beam
        left_beam = Polygon(
            self.grid["D2"] + LEFT*0.5, 
            self.grid["A3"] + RIGHT*0.5 + UP*0.5, 
            self.grid["F3"] + RIGHT*0.5 + DOWN*0.5,
            color="#00FFFF", fill_opacity=0.3, stroke_width=0
        )
        left_source = Star(n=5, color="#00FFFF", fill_opacity=1)
        self.place_at_grid(left_source, "D2", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(boundary), Write(boundary_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Show the "dim" puzzle
        self.play(FadeIn(right_puzzle), FadeIn(left_puzzle))
        self.play(left_puzzle.animate.set_opacity(0.05), right_puzzle.animate.set_opacity(0.05))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        self.play(FadeIn(right_source), Create(right_beam))
        self.play(right_puzzle.animate.set_color("#FFFF00").set_opacity(0.8))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#00FFFF")
        self.play(FadeIn(left_source), Create(left_beam))
        self.play(left_puzzle.animate.set_color("#00FFFF").set_opacity(0.8))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        # Merge everything into a seamless white image
        self.play(
            FadeOut(boundary),
            FadeOut(boundary_label),
            FadeOut(right_beam),
            FadeOut(left_beam),
            FadeOut(right_source),
            FadeOut(left_source),
            right_puzzle.animate.set_color(WHITE).set_opacity(1.0),
            left_puzzle.animate.set_color(WHITE).set_opacity(1.0)
        )
        
        # Add a glow to represent the "perfectly matched pattern"
        unified_puzzle = VGroup(left_puzzle, right_puzzle)
        self.play(unified_puzzle.animate.scale(1.05), rate_func=there_and_back)
        self.wait(2)
