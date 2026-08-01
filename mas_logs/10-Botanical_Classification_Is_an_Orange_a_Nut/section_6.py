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

class Section6Scene(TeachingScene):
    def construct(self):
        # Mandatory setup: title and lecture lines
        title_text = "The Final Verdict & Summary"
        lecture_lines = [
            "Pip enjoys the juice even if it's not crunchy.",
            "Oranges are berries, specifically a type called hesperidiums.",
            "Remember: nuts are dry, but oranges are fleshy."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight the first lecture line
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Assets: Pip the Squirrel and Orange
        # Using [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/squirrel.svg]
        pip = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/squirrel.svg").set_color("#D2691E")
        # Using [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/orange.svg]
        orange = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/orange.svg")
        
        # Simple straw construction
        straw = Line(ORIGIN, UP*0.8, color=WHITE, stroke_width=4).rotate(-PI/4)

        # Positioning using the visual anchor system
        self.place_at_grid(pip, "B2", scale_factor=0.6)
        self.place_at_grid(orange, "B4", scale_factor=0.6)
        self.place_at_grid(straw, "B3", scale_factor=0.8)

        self.play(FadeIn(pip), FadeIn(orange))
        self.play(Create(straw))

        # Sipping animation
        self.play(pip.animate.scale(1.15), rate_func=there_and_back, run_time=1.2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Classification text: 'Oranges = Hesperidium (a type of berry)'
        hesperidium_text = Text(
            "Oranges = Hesperidium\n(a type of berry)", 
            font_size=24, color="#FFA500", weight=BOLD
        )
        # Positioned in C1-C6 to bridge gap between assets and summary (Issue 46/47)
        self.place_in_area(hesperidium_text, "C1", "C6", scale_factor=0.8)

        self.play(Write(hesperidium_text))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Final text summary: 'Nuts are dry; Oranges are fleshy fruits.'
        summary_text = Text(
            "Nuts are dry fruits;\nOranges are fleshy fruits.", 
            font_size=24, color=WHITE
        )
        # Positioned in E1-E6 to avoid the very bottom edge (Issue 45)
        self.place_in_area(summary_text, "E1", "E6", scale_factor=0.8)

        self.play(FadeIn(summary_text))
        self.wait(3)

        # Cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
