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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Divergence: Source vs. Sink", [
            "Divergence measures expansion at a point.",
            "Positive divergence acts like a source.",
            "Negative divergence acts like a sink."
        ])
        
        # Load assets
        faucet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/faucet.svg")
        drain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drain.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.place_at_grid(faucet, 'C5', scale_factor=0.6)
        self.add(faucet)
        
        source_center = self.grid['C5']
        particles = VGroup(*[Dot(radius=0.03, color="#FF0000") for _ in range(30)])
        for p in particles:
            p.move_to(source_center)
        
        self.add(particles)
        self.play(
            *[p.animate.shift(0.8 * (np.array([np.cos(i), np.sin(i), 0]))) for i, p in enumerate(particles)],
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#0000FF")
        self.place_at_grid(drain, 'E5', scale_factor=0.6)
        self.add(drain)
        
        sink_center = self.grid['E5']
        sink_particles = VGroup(*[Dot(radius=0.03, color="#0000FF") for _ in range(30)])
        for p in sink_particles:
            angle = np.random.rand() * 2 * PI
            p.move_to(sink_center + 0.8 * np.array([np.cos(angle), np.sin(angle), 0]))
        
        self.add(sink_particles)
        self.play(
            *[p.animate.move_to(sink_center) for p in sink_particles],
            run_time=2
        )
        self.wait(2)
