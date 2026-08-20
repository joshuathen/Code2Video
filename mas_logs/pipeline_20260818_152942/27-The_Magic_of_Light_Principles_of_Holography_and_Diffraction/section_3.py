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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["The hologram acts as a diffraction grating.", "Shining the reference beam reconstructs wavefronts.", "This creates a true 3D image."]
        self.setup_layout("The Reconstruction: Diffraction in Action", lecture_lines)
        
        # Objects using Assets
        hologram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        beam = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        
        # Positioning based on feedback
        self.place_in_area(hologram, 'C3', 'D4', scale_factor=0.6)
        self.place_at_grid(beam, 'C3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(hologram), run_time=1)
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(beam), run_time=1)
        self.lecture[1].set_color("#FFD700")
        
        # Simulated diffraction waves
        waves = VGroup(*[Arc(radius=1+0.2*i, start_angle=-PI/4, angle=PI/2, color="#00FF00", stroke_opacity=0.6-0.1*i) for i in range(3)])
        waves.rotate(PI/2, about_point=ORIGIN)
        self.place_at_grid(waves, 'D4', scale_factor=0.9)
        self.play(FadeIn(waves), run_time=2)

        # === Animation for Lecture Line 3 ===
        ghost_apple = Circle(radius=0.5, color=GREEN, fill_opacity=0.3)
        self.place_at_grid(ghost_apple, 'D6', scale_factor=1.0)
        self.play(FadeIn(ghost_apple), run_time=2)
        self.lecture[2].set_color("#FFD700")
        
        self.wait(2)
