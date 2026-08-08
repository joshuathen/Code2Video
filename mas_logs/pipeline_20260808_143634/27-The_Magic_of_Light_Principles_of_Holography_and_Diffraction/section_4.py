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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Reconstruction: Decoding the Diffraction Pattern", [
            "Reconstruct by shining the reference beam.",
            "Light diffracts through the recorded pattern.",
            "Original wavefronts are faithfully recreated.",
            "A 3D image appears in space.",
            "The ghost image maintains spatial information."
        ])
        
        # Assets
        hologram_plate = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg")
        ghost_image = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        
        self.place_at_grid(hologram_plate, 'C2', scale_factor=0.6)
        
        beam = Arrow(start=LEFT*4, end=LEFT*0.5, color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        # Reconstruct by shining the reference beam.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(beam), run_time=2)
        self.play(FadeIn(hologram_plate))
        
        # === Animation for Lecture Line 2 ===
        # Light diffracts through the recorded pattern.
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        diffracted_light = VGroup(*[Line(start=hologram_plate.get_right(), end=hologram_plate.get_right()+RIGHT*2+UP*i*0.5, color=YELLOW, stroke_width=2) for i in range(-2, 3)])
        self.play(Create(diffracted_light))
        
        # === Animation for Lecture Line 3 ===
        # Original wavefronts are faithfully recreated.
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        wavefronts = VGroup(*[Arc(radius=0.5+i*0.5, angle=PI/2, start_angle=-PI/4, color=YELLOW) for i in range(3)])
        wavefronts.move_to(hologram_plate.get_right() + RIGHT*2)
        self.play(FadeIn(wavefronts))
        
        # === Animation for Lecture Line 4 ===
        # A 3D image appears in space.
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        self.place_at_grid(ghost_image, 'D5', scale_factor=0.8)
        self.play(FadeIn(ghost_image))
        
        # === Animation for Lecture Line 5 ===
        # The ghost image maintains spatial information.
        self.play(self.lecture[4].animate.set_color("#00FFFF"))
        self.play(ghost_image.animate.scale(1.2))
        self.wait(2)
