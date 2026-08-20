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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Key-Value Projection", [
            "MLP layers consist of two linear transformations.",
            "Non-linear activations bridge these linear projections.",
            "This architecture functions as a pattern matcher."
        ])

        # === Animation for Lecture Line 1 ===
        # MLP layers consist of two linear transformations.
        self.lecture[0].set_color("#FF5733")
        
        # W1 and W2 placeholders (using SVG asset if available, but the prompt says none.svg)
        w1 = Text("W1", color=WHITE)
        w2 = Text("W2", color=WHITE)
        
        self.place_at_grid(w1, 'B1', scale_factor=0.8)
        self.place_at_grid(w2, 'C1', scale_factor=0.8)
        self.play(FadeIn(w1), FadeIn(w2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Non-linear activations bridge these linear projections.
        self.lecture[1].set_color("#33FF57")
        sigma = MathTex(r"\sigma", color="#00FFFF")
        self.place_at_grid(sigma, 'B2', scale_factor=1.0)
        self.play(Write(sigma))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This architecture functions as a pattern matcher.
        self.lecture[2].set_color("#3357FF")
        
        # Visualizing GridConstellation asset
        # (Assuming the asset provides the structure, here we use dots for the pattern)
        constellation = VGroup(*[Dot(color="#FF00FF") for _ in range(8)])
        constellation.arrange_in_grid(rows=2, cols=4, buff=0.2)
        self.place_in_area(constellation, 'C4', 'F6', scale_factor=0.6)
        
        self.play(FadeIn(constellation))
        self.wait(2)
