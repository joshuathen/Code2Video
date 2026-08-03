from manim import *

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
        self.setup_layout("Reconstruction: Bringing the Ghost to Life", [
            "To see the image, we use the reference beam.",
            "Illuminating the hologram causes the light to diffract.",
            "The diffracted light waves perfectly mimic the original object.",
            "We see a three-dimensional \"ghost\" of the recorded object.",
            "The original wavefront is recreated, restoring depth and perspective."
        ])
        
        # Colors
        LASER_COLOR = "#FF0000"
        RAY_COLOR = "#87CEEB"
        GHOST_COLOR = "#F0FFF0"
        PLATE_COLOR = "#A9A9A9"

        # === Animation for Lecture Line 1 ===
        # To see the image, we use the reference beam.
        # A red laser #FF0000 [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg] shines through a holographic plate.
        
        self.lecture[0].set_color(LASER_COLOR)
        
        # Plate (Fixed position as per Issue 46)
        plate = Rectangle(width=0.2, height=4, color=PLATE_COLOR, fill_opacity=0.3)
        self.place_in_area(plate, 'B3', 'E3')
        
        # Laser Source (Asset Integration Issue 33)
        laser_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        laser_svg.set_color(LASER_COLOR)
        self.place_at_grid(laser_svg, "C1", scale_factor=0.6)
        
        # Laser Beam - starts from the laser icon to the plate center
        laser_beam = Line(laser_svg.get_center(), plate.get_center(), color=LASER_COLOR, stroke_width=4)
        
        self.play(Create(plate), FadeIn(laser_svg))
        self.play(Create(laser_beam))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Illuminating the hologram causes the light to diffract.
        # Blue light rays #87CEEB diffract and bend through the plate.
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RAY_COLOR)
        
        # Diffracted rays originating from the plate center
        diffracted_rays = VGroup(*[
            Line(plate.get_center(), self.grid[f"{row}6"], color=RAY_COLOR, stroke_width=2)
            for row in ["A", "B", "C", "D", "E"]
        ])
        
        self.play(Create(diffracted_rays))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The diffracted light waves perfectly mimic the original object.
        # The light rays reconstruct the original object's complex wavefront.
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RAY_COLOR)
        
        # Wavefront arcs expanding from the plate
        wavefronts = VGroup(*[
            Arc(radius=r, start_angle=-PI/3, angle=2*PI/3, color=RAY_COLOR, stroke_opacity=1-r/3)
            .move_to(plate.get_center() + RIGHT * (r/2))
            for r in [0.5, 1.0, 1.5]
        ])
        
        self.play(FadeIn(wavefronts, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # We see a three-dimensional "ghost" of the recorded object.
        # A shimmering virtual teapot #F0FFF0 [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/teapot.svg] appears in mid-air.
        
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GHOST_COLOR)
        
        # Teapot Asset (Asset Integration Issue 33, Position/Scale Issue 47)
        teapot_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/teapot.svg")
        teapot_svg.set_color(GHOST_COLOR)
        self.place_in_area(teapot_svg, 'C5', 'D6', scale_factor=0.6)
        
        # Shimmer effect using a ValueTracker and updater
        shimmer_tracker = ValueTracker(0)
        teapot_svg.add_updater(lambda m, dt: m.set_opacity(0.6 + 0.3 * np.sin(shimmer_tracker.get_value() * 5)))
        
        self.play(FadeIn(teapot_svg))
        self.play(shimmer_tracker.animate.set_value(2*PI), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The original wavefront is recreated, restoring depth and perspective.
        # The teapot #F0FFF0 [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/teapot.svg] rotates, revealing its three-dimensional structure.
        
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GHOST_COLOR)
        
        # Rotate around UP axis to simulate depth
        self.play(Rotate(teapot_svg, angle=TAU, axis=UP, run_time=3))
        self.wait(2)
