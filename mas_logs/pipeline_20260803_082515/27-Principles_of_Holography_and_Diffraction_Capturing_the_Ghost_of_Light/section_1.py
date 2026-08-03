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
        lecture_lines = [
            "Traditional photos record only the intensity of light waves.",
            "They capture a flat, two-dimensional view of the world.",
            "Holography captures both the intensity and the light's phase.",
            "This records the three-dimensional shape of the light wavefront.",
            "Moving your perspective reveals different angles of the object."
        ]
        self.setup_layout("The 3D Mystery: Photo vs. Hologram", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Traditional photos record only the intensity of light waves.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Camera at B2
        camera_body = Rectangle(width=0.6, height=0.4, color="#FFFFFF", fill_opacity=1)
        camera_lens = Circle(radius=0.1, color="#FFFFFF", fill_opacity=1).shift(UP*0.05)
        camera = VGroup(camera_body, camera_lens)
        self.place_at_grid(camera, "B2")
        
        # Cube (2D representation) at B4
        cube_obj = Square(side_length=0.6, color="#B0C4DE", fill_opacity=0.2)
        self.place_at_grid(cube_obj, "B4")
        
        self.play(FadeIn(camera), FadeIn(cube_obj))
        
        # Flash animation
        flash = Flash(self.grid["B4"], color=WHITE, line_length=0.2, num_lines=12)
        self.play(flash)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They capture a flat, two-dimensional view of the world.
        self.play(self.lecture[1].animate.set_color("#B0C4DE"))
        
        # 2D result at B6
        photo_rect = Square(side_length=0.6, color="#B0C4DE", fill_opacity=0.6)
        self.place_at_grid(photo_rect, "B6")
        label_2d = Text("2D Intensity Map", font_size=18, color="#B0C4DE")
        label_2d.next_to(photo_rect, DOWN, buff=0.1)
        
        conn_arrow = Arrow(self.grid["B4"], self.grid["B6"], buff=0.4, color="#B0C4DE")
        
        self.play(Create(conn_arrow), FadeIn(photo_rect), Write(label_2d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Holography captures both the intensity and the light's phase.
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Laser setup
        laser_box = Rectangle(width=0.5, height=0.3, color="#FF0000", fill_opacity=1)
        # Issue 34 Fix: D1 -> D2
        self.place_at_grid(laser_box, "D2")
        
        bs_mirror = Square(side_length=0.4, color=WHITE).rotate(45*DEGREES)
        self.place_at_grid(bs_mirror, "D3")
        
        # Adjust beam start: D2
        laser_beam_1 = Line(self.grid["D2"], self.grid["D3"], color="#FF0000", stroke_width=4)
        
        self.play(FadeIn(laser_box), FadeIn(bs_mirror))
        self.play(Create(laser_beam_1))
        
        # Issue 35 Fix: Plate at F4. Beams to F4.
        ref_beam = Line(self.grid["D3"], self.grid["F4"], color="#FF0000", stroke_width=3)
        obj_beam = Line(self.grid["D3"], self.grid["D5"], color="#FF0000", stroke_width=3)
        
        self.play(Create(ref_beam), Create(obj_beam))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This records the three-dimensional shape of the light wavefront.
        self.play(self.lecture[3].animate.set_color("#A9A9A9"))
        
        # Plate and wavefronts
        plate_rect = Rectangle(width=0.15, height=1.2, color="#A9A9A9", fill_opacity=1)
        # Issue 35 Fix: F3 -> F4
        self.place_at_grid(plate_rect, "F4")
        
        cube_3d_ref = Cube(side_length=0.7, stroke_color="#B0C4DE", fill_opacity=0.1)
        self.place_at_grid(cube_3d_ref, "D5")
        
        # Issue 35 Fix: Scattered beam to F4
        scattered_beam = Line(self.grid["D5"], self.grid["F4"], color="#FF0000", stroke_width=2)
        
        # Simple wavefront arcs
        wavefronts = VGroup(*[
            Arc(radius=0.2+i*0.2, start_angle=-60*DEGREES, angle=120*DEGREES, color="#A9A9A9")
            for i in range(3)
        ])
        wavefronts.move_to(self.grid["D5"])
        
        self.play(FadeIn(cube_3d_ref), FadeIn(plate_rect))
        self.play(Create(scattered_beam))
        # Move wavefronts towards F4
        self.play(wavefronts.animate.move_to(self.grid["F4"]).set_opacity(0), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Moving your perspective reveals different angles of the object.
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        
        # Issue 36 Fix: D4-F6 -> E5-F6
        hologram = Cube(side_length=1.0, stroke_color="#00FF00", fill_opacity=0.4)
        self.place_in_area(hologram, "E5", "F6")
        
        self.play(FadeIn(hologram))
        
        # Use an updater for continuous rotation to simulate changing perspective
        hologram.add_updater(lambda m, dt: m.rotate(dt * 0.8, axis=UP))
        hologram.add_updater(lambda m, dt: m.rotate(dt * 0.3, axis=RIGHT))
        
        self.wait(5)
        hologram.clear_updaters()
        self.wait(1)
