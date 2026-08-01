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
        # Data from storyboard
        title = "Recording the Hologram (The Interference Map)"
        lines = [
            "A laser beam is split into two separate paths.",
            "The reference beam travels directly to the holographic plate.",
            "The object beam reflects off the target onto plate.",
            "Both beams meet, creating a stationary interference pattern.",
            "This microscopic map encodes the 3D structure of light."
        ]
        
        # Setup layout
        self.setup_layout(title, lines)
        
        # Colors
        LASER_COLOR = "#00FFFF"
        PLATE_COLOR = "#CCCCCC"
        PATTERN_COLOR = "#FFFF00"
        OBJECT_COLOR = "#FF00FF"
        
        # Assets
        # Laser source [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg]
        laser_source = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color=WHITE)
        self.place_at_grid(laser_source, "A1", scale_factor=0.5)
        
        # Plate [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg]
        plate = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg", color=PLATE_COLOR)
        # Fix Issue 33: Move plate to area E6-F6
        self.place_in_area(plate, "E6", "F6", scale_factor=0.8)
        plate_label = Text("Holographic Plate", font_size=16, color=PLATE_COLOR)
        plate_label.next_to(plate, DOWN, buff=0.1)
        
        # Beam Splitter
        beam_splitter = Line(UP + LEFT, DOWN + RIGHT, color=BLUE_B).scale(0.3)
        self.place_at_grid(beam_splitter, "B2")
        
        # Mirror
        mirror = Line(LEFT, RIGHT, color=BLUE_B).scale(0.3).rotate(15 * DEGREES)
        # Fix Issue 34: Move mirror to A5
        self.place_at_grid(mirror, "A5")
        
        # Object
        # Using a star-like shape as the 3D object
        obj = Star(n=5, outer_radius=0.4, inner_radius=0.2, color=OBJECT_COLOR, fill_opacity=0.5)
        # Fix Issue 32: Move object to C5
        self.place_at_grid(obj, "C5")
        obj_label = Text("3D Object", font_size=16, color=OBJECT_COLOR)
        obj_label.next_to(obj, LEFT, buff=0.2)

        # === Animation for Lecture Line 1 ===
        # Line: "A laser beam is split into two separate paths."
        self.lecture[0].set_color(LASER_COLOR)
        
        main_beam = Line(self.grid["A1"], self.grid["B2"], color=LASER_COLOR)
        
        self.play(DrawBorderThenFill(laser_source), Create(beam_splitter))
        self.play(Create(main_beam))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Line: "The reference beam travels directly to the holographic plate."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(LASER_COLOR)
        
        ref_beam = Line(self.grid["B2"], plate.get_center(), color=LASER_COLOR)
        ref_label = Text("Reference Beam", font_size=14, color=LASER_COLOR)
        ref_label.next_to(ref_beam, UP, buff=-0.1).rotate(ref_beam.get_angle())
        
        self.play(DrawBorderThenFill(plate), Write(plate_label))
        self.play(Create(ref_beam), Write(ref_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "The object beam reflects off the target onto plate."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(LASER_COLOR)
        
        # Path: Splitter(B2) -> Mirror(A5) -> Object(C5) -> Plate
        path1 = Line(self.grid["B2"], self.grid["A5"], color=LASER_COLOR)
        path2 = Line(self.grid["A5"], self.grid["C5"], color=LASER_COLOR)
        path3 = Line(self.grid["C5"], plate.get_center(), color=LASER_COLOR)
        
        obj_beam_label = Text("Object Beam", font_size=14, color=LASER_COLOR)
        obj_beam_label.next_to(path1, UP, buff=0.1).rotate(path1.get_angle())

        self.play(Create(mirror), Create(obj), Write(obj_label))
        self.play(Create(path1), Write(obj_beam_label))
        self.play(Create(path2))
        self.play(Create(path3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line: "Both beams meet, creating a stationary interference pattern."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(PATTERN_COLOR)
        
        # Interference Pattern (dense series of lines to look microscopic)
        pattern = VGroup()
        rows_p, cols_p = 20, 20
        for i in range(rows_p):
            for j in range(cols_p):
                dot = Dot(
                    point=plate.get_center() + np.array([(j-cols_p/2)*0.03, (i-rows_p/2)*0.03, 0]),
                    radius=0.01,
                    color=PATTERN_COLOR,
                    fill_opacity=np.random.uniform(0.3, 1.0)
                )
                pattern.add(dot)
        
        self.play(FadeIn(pattern))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line: "This microscopic map encodes the 3D structure of light."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PATTERN_COLOR)
        
        # Visual flash to represent 'recording'
        flash = Flash(plate, color=PATTERN_COLOR, line_length=0.3, flash_radius=0.8)
        
        self.play(flash)
        # Final emphasis on the pattern - maybe a subtle glow or scale
        self.play(pattern.animate.scale(1.1).set_color(WHITE), run_time=0.5)
        self.play(pattern.animate.scale(1/1.1).set_color(PATTERN_COLOR), run_time=0.5)
        self.wait(2)
