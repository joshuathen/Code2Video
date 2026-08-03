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
        self.setup_layout("The Recording Process: Making the Blueprint", [
            "A laser beam is split into two separate paths.",
            "The object beam reflects off the subject being recorded.",
            "The reference beam travels directly to the holographic film.",
            "These beams collide, creating a complex interference pattern.",
            "This microscopic pattern encodes the object's 3D information."
        ])
        
        # Colors
        laser_color = "#FF0000"
        teapot_color = "#B0C4DE"
        fringe_color = "#E6E6FA"

        # === Animation for Lecture Line 1 ===
        # A red laser [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg] #FF0000 splits into two beams at a splitter.
        source = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color=laser_color)
        self.place_at_grid(source, "B2", scale_factor=0.6)
        source_label = Text("Laser", font_size=16).next_to(source, UP, buff=0.1)
        
        splitter = Rectangle(height=0.6, width=0.1, color=WHITE).rotate(45*DEGREES)
        self.place_at_grid(splitter, "B3")
        splitter_label = Text("Splitter", font_size=14).next_to(splitter, UP, buff=0.1)
        
        initial_beam = Line(self.grid["B2"], self.grid["B3"], color=laser_color)
        
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(source), FadeIn(source_label), FadeIn(splitter), FadeIn(splitter_label))
        self.play(Create(initial_beam))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # One beam reflects off a teapot [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/teapot.svg] #B0C4DE onto a plate.
        teapot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/teapot.svg", color=teapot_color)
        self.place_at_grid(teapot, "B5", scale_factor=0.6)
        teapot_label = Text("Teapot", font_size=16, color=teapot_color).next_to(teapot, UP, buff=0.1)
        
        plate = Rectangle(height=1.5, width=0.2, color=GREY_A, fill_opacity=0.3)
        self.place_at_grid(plate, "D5")
        plate_label = Text("Film", font_size=16).next_to(plate, DOWN, buff=0.1)
        
        object_beam_part1 = Line(self.grid["B3"], self.grid["B5"], color=laser_color)
        object_beam_part2 = Line(self.grid["B5"], self.grid["D5"], color=laser_color)
        obj_label = Text("Object Beam", font_size=12, color=laser_color).next_to(object_beam_part1, UP, buff=0.05)
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(teapot), FadeIn(teapot_label), FadeIn(plate), FadeIn(plate_label))
        self.play(Create(object_beam_part1), FadeIn(obj_label))
        self.play(Create(object_beam_part2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A second beam travels directly to the plate as a reference.
        mirror = Rectangle(height=0.4, width=0.05, color=WHITE).rotate(45*DEGREES)
        self.place_at_grid(mirror, "D3")
        mirror_label = Text("Mirror", font_size=14).next_to(mirror, LEFT, buff=0.1)
        
        ref_beam_part1 = Line(self.grid["B3"], self.grid["D3"], color=laser_color)
        ref_beam_part2 = Line(self.grid["D3"], self.grid["D5"], color=laser_color)
        ref_label = Text("Reference Beam", font_size=12, color=laser_color).next_to(ref_beam_part1, LEFT, buff=0.05)
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(FadeIn(mirror), FadeIn(mirror_label))
        self.play(Create(ref_beam_part1))
        self.play(Create(ref_beam_part2), FadeIn(ref_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Both beams meet, forming an interference pattern #E6E6FA on the plate.
        fringes = VGroup(*[
            Line(UP*0.7, DOWN*0.7, color=fringe_color, stroke_width=0.8).shift(RIGHT * x)
            for x in np.linspace(-0.06, 0.06, 12)
        ])
        fringes.move_to(plate.get_center())
        
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(FadeIn(fringes))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Zoom into the plate showing microscopic details of the pattern.
        zoom_pattern = VGroup()
        for i in range(25):
            x_off = -2.5 + i * 0.2
            pts = [np.array([x_off + 0.1 * np.sin(y * 3 + i), y, 0]) for y in np.linspace(-2.5, 2.5, 30)]
            line = VMobject(color=fringe_color, stroke_width=1.5)
            line.set_points_as_corners(pts)
            zoom_pattern.add(line)
        
        self.place_in_area(zoom_pattern, "A1", "F6", scale_factor=1.0)
        
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        setup_group = VGroup(source, source_label, splitter, splitter_label, initial_beam, 
                            teapot, teapot_label, plate, plate_label, object_beam_part1, object_beam_part2,
                            obj_label, mirror, mirror_label, ref_beam_part1, ref_beam_part2, ref_label, fringes)
        
        self.play(
            setup_group.animate.set_opacity(0),
            FadeIn(zoom_pattern)
        )
        self.wait(3)
        
        self.lecture[4].set_color(WHITE)
        self.wait(1)
