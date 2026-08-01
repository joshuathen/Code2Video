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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout("The Squirrel's Dilemma (Introduction)", [
            "Meet Pip, a squirrel who loves crunchy nuts.",
            "He finds a mysterious, bright orange sphere.",
            "Instead of a crunch, he gets a squishy splash!"
        ])

        # === Animation for Lecture Line 1 ===
        # Create Tree (#8B4513 trunk, #228B22 leaves)
        trunk = Rectangle(height=2.5, width=0.6, color="#8B4513", fill_opacity=1)
        leaves = Circle(radius=1.2, color="#228B22", fill_opacity=0.8)
        tree = VGroup(trunk, leaves.shift(UP * 1.5))
        
        # [VideoCritic] Fix (Issue 34): Place tree in area B2 to E3
        self.place_in_area(tree, 'B2', 'E3', scale_factor=0.9)

        # Create Pip using asset: [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/squirrel.svg]
        pip = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/squirrel.svg")
        pip.set_color("#D2691E")
        
        # [VideoCritic] Fix (Issue 35): Place Pip at grid D4
        self.place_at_grid(pip, 'D4', scale_factor=0.8)

        # Highlight lecture line 1 and display Pip and Tree
        self.play(self.lecture[0].animate.set_color("#D2691E"))
        self.play(FadeIn(tree), FadeIn(pip))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create Orange using asset: [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/orange.svg]
        orange = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/orange.svg")
        orange.set_color("#FFA500")
        
        # [VideoCritic] Fix (Issue 36): Place Orange at grid D5
        self.place_at_grid(orange, 'D5', scale_factor=0.6)

        # Highlight next lecture line and roll orange toward Pip
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFA500")
        )
        # Move orange slightly left toward Pip
        self.play(orange.animate.shift(LEFT * 0.4), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight final line with juice color
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )

        # Create yellow juice particles (#FFFF00)
        juice_particles = VGroup(*[
            Dot(color="#FFFF00", radius=0.05).move_to(orange.get_center()) for _ in range(12)
        ])

        # Interaction: Pip tries to crack the orange (moves slightly), orange squishes, juice sprays
        self.play(pip.animate.shift(RIGHT * 0.1), run_time=0.3)
        self.play(
            orange.animate.stretch(0.4, dim=1).stretch(1.3, dim=0),
            LaggedStart(
                *[p.animate.shift(
                    np.array([np.cos(i * TAU/12), np.sin(i * TAU/12), 0]) * 1.5
                ).set_opacity(0) for i, p in enumerate(juice_particles)],
                lag_ratio=0.05
            ),
            run_time=1.0
        )

        # Reset text color at end of section
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
