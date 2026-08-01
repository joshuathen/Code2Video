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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "The Reconstruction (Bringing the Ghost to Life)"
        lecture_lines = [
            "Reconstruct the image by shining the reference beam again.",
            "The developed plate's pattern diffracts the incoming light.",
            "Light rays bend according to the diffraction equation.",
            "This process regenerates the original object's wavefront in 3D.",
            "The viewer sees the object as if it were present."
        ]
        self.setup_layout(title, lecture_lines)

        # Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg]
        fish_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg", color="#00FF00", fill_opacity=0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Holographic Plate
        plate = Rectangle(height=2.5, width=0.2, color=WHITE, fill_opacity=0.3, fill_color=WHITE)
        # Adding some "pattern" to the plate to represent the interference map
        pattern = VGroup(*[Line(LEFT*0.1, RIGHT*0.1, stroke_width=1, color=GREY_B).shift(UP*i*0.2) for i in range(-5, 6)])
        hologram_plate = VGroup(plate, pattern)
        self.place_in_area(hologram_plate, "B3", "E3")
        
        # Reference Beam Source (imaginary laser point)
        beam_source_pos = self.grid["C1"]
        
        # Reference Beam Arrow
        beam_arrow = Arrow(beam_source_pos, self.grid["C3"], color="#00FFFF", buff=0)
        beam_label = Text("Reference Beam", font_size=16, color="#00FFFF").next_to(beam_arrow, UP, buff=0.1)
        
        self.play(FadeIn(hologram_plate))
        self.play(Create(beam_arrow), Write(beam_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Diffracted light rays coming out of the plate
        diffracted_rays = VGroup()
        for angle in [-30, -15, 0, 15, 30]:
            start = self.grid["C3"]
            # Rays spread out to the right
            end = start + 2.5 * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0])
            ray = Line(start, end, color="#00FFFF", stroke_width=2, stroke_opacity=0.6)
            diffracted_rays.add(ray)
            
        self.play(Create(diffracted_rays))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # The diffraction equation
        diffraction_eq = MathTex("n\\lambda = d \\sin\\theta", font_size=36, color=YELLOW)
        # Issue 35 fix: Move to A5 and scale 0.8
        self.place_at_grid(diffraction_eq, "A5", scale_factor=0.8)
        
        self.play(Write(diffraction_eq))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # 3D ghost-like fish reconstruction using SVG asset
        self.place_in_area(fish_svg, "C4", "E5", scale_factor=1.0)
        
        label = Text("Wavefront Reconstruction", font_size=18, color="#00FF00")
        # Issue 36 fix: Place in area F4-F5 and scale 0.8
        self.place_in_area(label, "F4", "F5", scale_factor=0.8)
        
        # Ghostly glow effect
        fish_glow = fish_svg.copy().scale(1.1).set_style(stroke_width=6, stroke_opacity=0.2, stroke_color="#00FF00")
        reconstructed_fish = VGroup(fish_svg, fish_glow)
        
        self.play(FadeIn(reconstructed_fish, shift=RIGHT), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Viewer seeing the fish (eye mobject)
        eye_upper = Arc(radius=0.3, start_angle=PI/6, angle=2*PI/3, color=WHITE)
        eye_lower = Arc(radius=0.3, start_angle=7*PI/6, angle=2*PI/3, color=WHITE)
        pupil = Dot(radius=0.05, color=WHITE)
        eye = VGroup(eye_upper, eye_lower, pupil)
        # Issue 37 fix: Scale eye at D6 by 0.7
        self.place_at_grid(eye, "D6", scale_factor=0.7)
        
        # Sight lines from eye to the fish
        sight_lines = VGroup(
            Line(self.grid["D6"], fish_svg.get_center() + UP*0.3, stroke_width=1, color=WHITE, stroke_opacity=0.3),
            Line(self.grid["D6"], fish_svg.get_center() - UP*0.3, stroke_width=1, color=WHITE, stroke_opacity=0.3)
        )
        
        self.play(Create(eye), Create(sight_lines))
        self.wait(3)
