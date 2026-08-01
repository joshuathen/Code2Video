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
        self.setup_layout("The Limitation: Detection vs. Correction", [
            "Parity tells us an error exists.",
            "But it cannot show which bit is wrong.",
            "To fix data, we need more clever guards."
        ])

        # Colors
        SENT_COLOR = BLUE_C
        RCVD_COLOR = RED_C
        TEXT_COLOR = WHITE

        # Setup persistent time tracker for updaters
        self.time_tracker = ValueTracker(0)
        self.add(self.time_tracker)
        def update_time(dt):
            self.time_tracker.increment_value(dt)
        self.add_updater(update_time)

        # === Animation for Lecture Line 1 ===
        # Parity tells us an error exists.
        self.lecture[0].set_color(YELLOW)
        
        sent_label = Text("SENT:", font_size=24, color=SENT_COLOR)
        self.place_at_grid(sent_label, "A1")
        
        sent_bits = VGroup(*[Text(b, font_size=36, color=SENT_COLOR) for b in "1010"]).arrange(RIGHT, buff=0.4)
        self.place_in_area(sent_bits, "A2", "A5")
        
        rcvd_label = Text("RCVD:", font_size=24, color=RCVD_COLOR)
        self.place_at_grid(rcvd_label, "C1")
        
        rcvd_bits = VGroup(*[Text(b, font_size=36, color=RCVD_COLOR) for b in "1000"]).arrange(RIGHT, buff=0.4)
        self.place_in_area(rcvd_bits, "C2", "C5")
        
        self.play(Write(sent_label), Write(sent_bits))
        self.wait(1)
        self.play(Write(rcvd_label), Write(rcvd_bits))
        
        # Parity check logic
        parity_note = Text("Sum = 1 (Odd) -> ERROR!", font_size=20, color=RED)
        # Resolved Issue 32: Positioning and Scaling
        self.place_in_area(parity_note, "D2", "D6", scale_factor=0.7)
        self.play(FadeIn(parity_note))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # But it cannot show which bit is wrong.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Resolved Issue 23: Asset Integration
        # Siren icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/siren.svg]
        siren = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/siren.svg")
        siren.set_color(RED)
        self.place_in_area(siren, "C2", "C5", scale_factor=1.2)
        
        def siren_updater(m):
            t = self.time_tracker.get_value()
            m.set_opacity(0.4 * np.sin(t * 10) + 0.6)

        siren.add_updater(siren_updater)
        self.play(FadeIn(siren))
        
        question_text = Text("Which bit is the liar?", font_size=24, color=YELLOW)
        # Resolved Issue 33: Positioning and Scaling
        self.place_in_area(question_text, "E2", "E6", scale_factor=0.7)
        
        # Highlight bits
        highlights = VGroup(*[SurroundingRectangle(b, color=YELLOW, buff=0.1) for b in rcvd_bits])
        
        self.play(Write(question_text))
        self.play(Create(highlights))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # To fix data, we need more clever guards.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        guards_text = Text("Need 'Guards' in specific spots...", font_size=22, color=BLUE_A)
        # Resolved Issue 34: Positioning and Scaling
        self.place_in_area(guards_text, "F2", "F6", scale_factor=0.6)
        
        self.play(FadeIn(guards_text))
        self.wait(2)
        
        siren.remove_updater(siren_updater)
        self.play(
            FadeOut(siren), 
            FadeOut(highlights), 
            FadeOut(question_text), 
            FadeOut(guards_text), 
            FadeOut(parity_note),
            FadeOut(sent_label),
            FadeOut(sent_bits),
            FadeOut(rcvd_label),
            FadeOut(rcvd_bits)
        )
        self.wait(1)
