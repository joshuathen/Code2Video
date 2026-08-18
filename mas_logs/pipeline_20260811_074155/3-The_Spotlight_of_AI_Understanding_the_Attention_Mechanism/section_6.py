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
        title = "The Result: Contextualized Understanding"
        lines = [
            "Weighted values combine into a new, context-aware vector.",
            "This vector captures the word's current meaning perfectly.",
            "Contextual understanding allows for more human-like AI.",
            "Models can now predict the next logical word.",
            "This is the secret behind powerful language models."
        ]
        self.setup_layout(title, lines)

        # Colors
        NEUTRAL_GRAY = "#808080"
        CONTEXT_BLUE = "#1E90FF"
        HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT))
        
        # Show the word 'Bank' represented as a neutral gray vector
        bank_arrow = Arrow(start=ORIGIN, end=UP*1.2, color=NEUTRAL_GRAY, buff=0)
        bank_label = Text("Bank", font_size=24, color=NEUTRAL_GRAY).next_to(bank_arrow, DOWN, buff=0.1)
        bank_group = VGroup(bank_arrow, bank_label)
        self.place_in_area(bank_group, "C3", "D4")
        
        self.play(Create(bank_arrow), Write(bank_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT)
        )

        # Asset for River [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg]
        river_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/river.svg").set_color(CONTEXT_BLUE).scale(0.3)
        river_arrow = Arrow(start=ORIGIN, end=UP*0.8, color=CONTEXT_BLUE, buff=0)
        river_label = Text("River", font_size=20, color=CONTEXT_BLUE)
        river_group = VGroup(river_icon, river_arrow, river_label).arrange(DOWN, buff=0.1)
        
        # Fixing Issue 33: place at B3 instead of B2
        self.place_at_grid(river_group, "B3")

        water_arrow = Arrow(start=ORIGIN, end=UP*0.8, color=CONTEXT_BLUE, buff=0)
        water_label = Text("Water", font_size=20, color=CONTEXT_BLUE).next_to(water_arrow, DOWN, buff=0.1)
        water_group = VGroup(water_arrow, water_label)
        
        # Fixing Issue 34: place at E4 instead of E5
        self.place_at_grid(water_group, "E4")

        self.play(FadeIn(river_group), FadeIn(water_group))
        
        # Drifting animation toward the center (Bank)
        self.play(
            river_group.animate.move_to(bank_group.get_center() + LEFT*0.5 + UP*0.5).set_opacity(0.3),
            water_group.animate.move_to(bank_group.get_center() + RIGHT*0.5 + DOWN*0.5).set_opacity(0.3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT)
        )

        # Animate the 'Bank' vector gradually turning blue
        self.play(
            bank_arrow.animate.set_color(CONTEXT_BLUE),
            bank_label.animate.set_color(CONTEXT_BLUE),
            FadeOut(river_group),
            FadeOut(water_group),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT)
        )

        # Predict the next word: "Flowing"
        prediction_box = Rectangle(width=2.5, height=0.8, color=WHITE)
        prediction_text = Text("Next: Flowing", font_size=24, color=CONTEXT_BLUE)
        prediction_group = VGroup(prediction_box, prediction_text)
        
        # Fixing Issue 32: place in area F4-F6 instead of at grid F4
        self.place_in_area(prediction_group, "F4", "F6")

        self.play(Create(prediction_box), Write(prediction_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT)
        )
        
        # Final emphasis
        glow = bank_arrow.copy().set_stroke(width=10).set_color(CONTEXT_BLUE).set_opacity(0.4)
        self.play(FadeIn(glow), bank_group.animate.scale(1.1))
        self.play(FadeOut(glow))
        self.wait(2)

        # Final cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
