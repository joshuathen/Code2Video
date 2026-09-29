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
        self.setup_layout("Curl: The Rotation Measure", [
            "Curl measures rotation around a point.",
            "Think of a small spinning pinwheel.",
            "Circulation is non-zero in swirling fields."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Particles rotating (visualizing curl) with pinwheel asset
        center = self.grid['C4']
        pinwheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pinwheel.svg")
        self.place_at_grid(pinwheel, 'C4', scale_factor=0.5)
        pinwheel.set_color("#00FF00")
        self.add(pinwheel)
        
        particles = VGroup(*[Dot(color="#00FF00", radius=0.05).move_to(center + rotate_vector(RIGHT * 1.5, angle)) for angle in np.linspace(0, 2*PI, 12, endpoint=False)])
        self.add(particles)
        
        def update_particles(mob, dt):
            mob.rotate(dt * 0.5, about_point=center)
            pinwheel.rotate(dt * 0.5)
        particles.add_updater(update_particles)
        self.wait(2)
        particles.remove_updater(update_particles)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Using pinwheel already placed
        self.play(Rotate(pinwheel, angle=2*PI, run_time=2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        arc = Arc(start_angle=0, angle=1.5*PI, radius=0.8, color=WHITE, arc_center=center)
        arrow = Arrow(arc.get_start(), arc.get_end(), buff=0, color=WHITE).scale(0.3)
        self.play(Create(arc), Create(arrow))
        self.wait(2)
