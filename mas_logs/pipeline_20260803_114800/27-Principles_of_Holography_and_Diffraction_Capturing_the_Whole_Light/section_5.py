from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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

class Section5Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "The Reconstruction Process: Diffraction at Work"
        lines = [
            "Re-illuminate the plate with the laser.",
            "Light diffracts through the fringe pattern.",
            "This process reconstructs the original wavefront.",
            "The object appears in 3D space.",
            "It seems to float behind the plate."
        ]
        self.setup_layout(title, lines)
        
        # === Elements Construction ===
        
        # 1. Developed Plate (at C5)
        plate_rect = Rectangle(height=3.0, width=0.15, color=GRAY, fill_opacity=0.2)
        # Internal pattern to represent fringes
        fringes = VGroup(*[
            Line(UP*1.4, DOWN*1.4, stroke_width=0.5, color=GREY_A).shift(RIGHT * (i * 0.015)) 
            for i in range(-4, 5)
        ])
        plate = VGroup(plate_rect, fringes)
        self.place_at_grid(plate, 'C5', scale_factor=1.2)
        
        # 2. Laser Source at B1 - Procedural replacement for missing SVG asset
        laser_body = Rectangle(width=0.8, height=0.3, fill_opacity=1, color=DARK_GREY)
        laser_tip = Triangle(fill_opacity=1, color=RED).scale(0.1).rotate(-PI/2).next_to(laser_body, RIGHT, buff=0)
        laser = VGroup(laser_body, laser_tip)
        laser.set_color("#FF0000")
        self.place_at_grid(laser, 'B1', scale_factor=0.8)
        # Adjust orientation
        laser.rotate(angle_of_vector(self.grid['C5'] - self.grid['B1']))
        
        # 3. Chess Piece at C2 (Virtual Image) - Procedural replacement for missing SVG asset
        chess_piece = VGroup(
            Circle(radius=0.25, fill_opacity=1),
            Polygon(
                [-0.2, -0.4, 0], [0.2, -0.4, 0], [0.15, 0, 0], [-0.15, 0, 0],
                fill_opacity=1
            )
        ).set_color("#C0C0C0")
        chess_piece.set_opacity(0)
        self.place_at_grid(chess_piece, 'C2', scale_factor=0.8)
        
        # 4. Reference Beam (Path from B1 to Plate C5)
        beam = Polygon(
            self.grid['B1'], 
            self.grid['C5'] + UP*0.8 + LEFT*0.05, 
            self.grid['C5'] + DOWN*0.8 + LEFT*0.05,
            color=RED, fill_opacity=0.3, stroke_width=0
        )
        
        # 5. Reconstructed Wavefront (Diverging rays towards viewer on the right)
        targets = ['B6', 'C6', 'D6']
        real_rays = VGroup(*[
            Line(self.grid['C5'], self.grid[t], color=RED, stroke_width=2, stroke_opacity=0.6)
            for t in targets
        ])
        
        # 6. Virtual Wavefront (Dashed rays connecting virtual object C2 to plate C5)
        virtual_targets = [UP*0.5, ORIGIN, DOWN*0.5]
        virtual_rays = VGroup(*[
            DashedLine(self.grid['C2'], self.grid['C5'] + v, color=RED, stroke_width=1, stroke_opacity=0.4)
            for v in virtual_targets
        ])

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(plate), FadeIn(laser))
        self.play(DrawBorderThenFill(beam))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(fringes.animate.set_stroke(color=WHITE, width=1.5), run_time=0.5)
        self.play(fringes.animate.set_stroke(color=GREY_A, width=0.5), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(LaggedStart(*[Create(ray) for ray in real_rays], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(chess_piece.animate.set_opacity(0.6), run_time=2)
        flash = Flash(chess_piece, color=WHITE, line_length=0.2, flash_radius=0.4)
        self.play(flash)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(Create(virtual_rays))
        self.play(
            chess_piece.animate.shift(UP * 0.15),
            run_time=1.5,
            rate_func=there_and_back
        )
        self.wait(2)
