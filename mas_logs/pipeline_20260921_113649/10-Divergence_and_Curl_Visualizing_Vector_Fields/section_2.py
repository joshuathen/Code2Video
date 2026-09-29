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
        self.setup_layout("Divergence: The 'Source/Sink' Concept", [
            "Divergence measures field spreading or convergence.",
            "Positive divergence indicates a source point.",
            "Negative divergence signifies a sink point."
        ])
        
        # Vector Field Visualization
        # Use ArrowVectorField for VGroup of vectors, as VectorField isn't a direct mobject with these args
        field = ArrowVectorField(lambda pos: pos * 0.2, x_range=[-3, 3, 1], y_range=[-3, 3, 1])
        self.place_in_area(field, 'B3', 'E6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(field))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        source_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/faucet.svg")
        source_label = Text("Source (+)", color="#00FF00")
        group_s = VGroup(source_icon, source_label).arrange(DOWN)
        self.place_at_grid(group_s, 'A5', scale_factor=0.8)
        self.play(FadeIn(group_s))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        sink_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drain.svg")
        sink_label = Text("Sink (-)", color="#FF0000")
        group_d = VGroup(sink_icon, sink_label).arrange(DOWN)
        self.place_at_grid(group_d, 'F2', scale_factor=0.8)
        self.play(FadeIn(group_d))
        self.lecture[2].set_color("#FF0000")
        
        self.wait(2)
