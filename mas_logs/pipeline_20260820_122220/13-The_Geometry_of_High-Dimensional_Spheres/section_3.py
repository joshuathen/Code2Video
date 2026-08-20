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
        lecture_lines = [
            "Volume concentrates near the surface in high dimensions.",
            "Most of the sphere's mass lies near the crust.",
            "This behavior is counter-intuitive for low-dimensional intuition.",
            "Visualizing this, the crust dominates the volume.",
            "Volume concentration is a fundamental geometric fact."
        ]
        self.setup_layout("The Paradox of Concentration of Measure", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Using SVG asset
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere.set_color("#BDC3C7")
        self.place_in_area(sphere, "A3", "C5", scale_factor=0.7)
        self.play(Create(sphere))
        self.lecture[0].set_color("#BDC3C7")

        # === Animation for Lecture Line 2 ===
        # Use a ring to represent the crust
        crust = Annulus(inner_radius=0.8, outer_radius=1.0, color="#E67E22", fill_opacity=0.6)
        self.place_in_area(crust, "A3", "C5", scale_factor=0.7)
        self.play(FadeIn(crust))
        self.lecture[1].set_color("#E67E22")

        # === Animation for Lecture Line 3 ===
        # Fade in distribution curve showing concentration of points (#9B59B6).
        curve = FunctionGraph(lambda x: 0.5 * np.exp(-x**2 * 5), x_range=[-1.5, 1.5], color="#9B59B6")
        self.place_in_area(curve, "E3", "F5", scale_factor=0.6)
        self.play(Create(curve))
        self.lecture[2].set_color("#9B59B6")

        # === Animation for Lecture Line 4 ===
        # Display text 'Majority Volume' (#E67E22) and anchor it.
        label = Text("Majority Volume", font_size=20, color="#E67E22")
        self.place_at_grid(label, "A4", scale_factor=0.7)
        self.play(Write(label))
        self.lecture[3].set_color("#FFFFFF")

        # === Animation for Lecture Line 5 ===
        # Flash the crust area to emphasize concentration, color #E74C3C.
        self.play(Indicate(crust, color="#E74C3C", scale_factor=1.1))
        self.lecture[4].set_color("#E74C3C")
        self.wait(1)
