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
        lecture_lines = ["Divergence measures spreading at a point.", "Positive divergence acts as a source.", "Negative divergence acts as a sink.", "Regions expand or contract accordingly.", "It quantifies net flow behavior."]
        self.setup_layout("Divergence: Source and Sink", lecture_lines)
        
        # Helper to create source/sink vector fields
        def get_field(type="source"):
            return ArrowVectorField(
                lambda p: (p / np.linalg.norm(p)) if (type == "source" and np.linalg.norm(p) > 0.1) else (-(p / np.linalg.norm(p)) if (type == "sink" and np.linalg.norm(p) > 0.1) else np.array([0, 0, 0])),
                x_range=[-2, 2, 0.5],
                y_range=[-2, 2, 0.5],
            )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GRAY)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        source_field = get_field("source").set_color("#FF0000")
        faucet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/faucet.svg").set_color("#FF0000")
        self.place_in_area(source_field, 'A4', 'B6', scale_factor=0.6)
        self.place_at_grid(faucet, 'A5', scale_factor=0.3)
        self.play(Create(source_field), FadeIn(faucet))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#0000FF")
        sink_field = get_field("sink").set_color("#0000FF")
        drain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drain.svg").set_color("#0000FF")
        self.place_in_area(sink_field, 'D4', 'E6', scale_factor=0.6)
        self.place_at_grid(drain, 'D5', scale_factor=0.3)
        self.play(Create(sink_field), FadeIn(drain))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        formula = MathTex(r"\nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y}", color="#FFFF00")
        self.place_at_grid(formula, 'E5', scale_factor=0.9)
        self.play(Write(formula))
        self.wait(2)
