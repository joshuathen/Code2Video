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
        # Data from storyboard
        title = "The Recording Process: Creating the Interference Pattern"
        lines = [
            "A beam splitter divides the laser.",
            "The reference beam hits the plate.",
            "The object beam reflects off subject.",
            "They meet to form interference fringes.",
            "This fingerprint stores the 3D data."
        ]
        self.setup_layout(title, lines)

        # Colors
        LASER_COLOR = "#FF0000"
        CHESS_COLOR = "#C0C0C0"
        SPLITTER_COLOR = "#ADD8E6"
        PLATE_COLOR = "#FFFFFF"
        FRINGE_COLOR = "#FFFF00"
        DATA_COLOR = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # "A beam splitter divides the laser."
        self.lecture[0].set_color(LASER_COLOR)
        
        laser_source = Square(side_length=0.4, color=GRAY, fill_opacity=1)
        self.place_at_grid(laser_source, "C1")
        source_label = Text("Laser", font_size=16, color=WHITE)
        self.place_at_grid(source_label, "B1")
        
        splitter = Rectangle(height=1.0, width=0.1, color=SPLITTER_COLOR, fill_opacity=0.5).rotate(45*DEGREES)
        self.place_at_grid(splitter, "C3")
        splitter_label = Text("Splitter", font_size=16, color=SPLITTER_COLOR)
        # Resolved Issue 39: self.place_in_area(splitter_label, 'B2', 'B4', scale_factor=0.6)
        self.place_in_area(splitter_label, 'B2', 'B4', scale_factor=0.6)
        
        main_beam = Line(self.grid["C1"], self.grid["C3"], color=LASER_COLOR, stroke_width=4)
        
        self.play(Create(laser_source), FadeIn(source_label))
        self.play(Create(splitter), FadeIn(splitter_label))
        self.play(Create(main_beam))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The reference beam hits the plate."
        self.lecture[1].set_color(PLATE_COLOR)
        
        plate = Rectangle(height=2.5, width=0.2, color=PLATE_COLOR, fill_opacity=0.3)
        self.place_in_area(plate, "C6", "E6")
        plate_label = Text("Plate", font_size=16, color=PLATE_COLOR)
        self.place_at_grid(plate_label, "F6")
        
        ref_beam = Line(self.grid["C3"], self.grid["C6"], color=LASER_COLOR, stroke_width=4)
        ref_label = Text("Reference Beam", font_size=14, color=LASER_COLOR)
        # Resolved Issue 37: self.place_in_area(ref_label, 'B4', 'B6', scale_factor=0.6)
        self.place_in_area(ref_label, 'B4', 'B6', scale_factor=0.6)
        
        self.play(Create(plate), FadeIn(plate_label))
        self.play(Create(ref_beam), FadeIn(ref_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The object beam reflects off subject."
        self.lecture[2].set_color(CHESS_COLOR)
        
        # Simple chess piece (Pawn)
        chess_piece = VGroup(
            Circle(radius=0.15, color=CHESS_COLOR, fill_opacity=1),
            Triangle(color=CHESS_COLOR, fill_opacity=1).scale(0.4).shift(DOWN*0.3)
        )
        self.place_at_grid(chess_piece, "E3")
        chess_label = Text("Object", font_size=16, color=CHESS_COLOR)
        self.place_at_grid(chess_label, "F3")
        
        obj_beam = Line(self.grid["C3"], self.grid["E3"], color=LASER_COLOR, stroke_width=4)
        obj_label = Text("Object Beam", font_size=14, color=LASER_COLOR)
        # Resolved Issue 38: self.place_in_area(obj_label, 'D1', 'D2', scale_factor=0.6)
        self.place_in_area(obj_label, 'D1', 'D2', scale_factor=0.6)
        
        # Scattering beams from E3 to C6, D6, E6
        scatter_paths = ["C6", "D6", "E6"]
        scatter_beams = VGroup(*[
            Line(self.grid["E3"], self.grid[pos], color=LASER_COLOR, stroke_width=1, stroke_opacity=0.5)
            for pos in scatter_paths
        ])

        self.play(Create(chess_piece), FadeIn(chess_label))
        self.play(Create(obj_beam), FadeIn(obj_label))
        self.play(Create(scatter_beams))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "They meet to form interference fringes."
        self.lecture[3].set_color(FRINGE_COLOR)
        
        # Draw some yellow vertical lines on the plate to represent fringes
        fringes = VGroup(*[
            Line(UP*1.0, DOWN*1.0, color=FRINGE_COLOR, stroke_width=2).shift(RIGHT * (5.5 + x * 0.08))
            for x in range(-5, 6)
        ])
        # Place at plate center (D6)
        fringes.move_to(self.grid["D6"])
        
        self.play(FadeIn(fringes))
        self.play(Indicate(fringes, color=FRINGE_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This fingerprint stores the 3D data."
        self.lecture[4].set_color(DATA_COLOR)
        
        # Noise pattern: random tiny dots
        noise = VGroup()
        for _ in range(150):
            d = Dot(radius=0.015, color=DATA_COLOR)
            # Random position within the plate area
            d.move_to(self.grid["D6"] + np.array([
                np.random.uniform(-0.1, 0.1),
                np.random.uniform(-1.2, 1.2),
                0
            ]))
            noise.add(d)
        
        # Transition to noise and "zoom"
        self.play(
            FadeOut(fringes),
            FadeIn(noise),
            plate.animate.scale(1.2),
            noise.animate.scale(1.2)
        )
        self.wait(2)
