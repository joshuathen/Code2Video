from manim import *
import numpy as np

SILVER = "#C0C0C0"

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
        # Setup layout with title and lecture lines
        lecture_lines = [
            'Classical objects exist in one definite state at once.',
            'But quantum states can merge into a single blur.',
            'This combined state is known as quantum superposition.',
            'A clear boundary separates the classical and quantum worlds.',
            'The quantum side pulses with all possible outcomes.'
        ]
        self.setup_layout("The Classical vs. Quantum Worldview", lecture_lines)

        # Asset path
        COIN_PATH = "/mmfs1/data/home/jthen/Code2Video/assets/icon/coin.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Heads coin at B2 (Fix: moved from C2 per Issue 40)
        heads_coin = SVGMobject(COIN_PATH).set_color("#FFD700").set_opacity(0.8)
        heads_label = Text("Heads", font_size=16, color="#FFD700")
        heads_group = VGroup(heads_coin, heads_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(heads_group, "B2", scale_factor=0.6)

        # Tails coin at D2 (Fix: moved from E2 per Issue 41)
        tails_coin = SVGMobject(COIN_PATH).set_color(SILVER).set_opacity(0.8)
        tails_label = Text("Tails", font_size=16, color=SILVER)
        tails_group = VGroup(tails_coin, tails_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(tails_group, "D2", scale_factor=0.6)

        self.play(FadeIn(heads_group), FadeIn(tails_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#888888")
        )

        # Merge coins and spin into a grey blur at D5
        blur_coin = SVGMobject(COIN_PATH).set_color("#888888").set_opacity(0.6)
        # Add rotation indicators for visual spin effect
        spin_indicators = VGroup(*[
            Line(ORIGIN, UP*0.5, stroke_width=2, color="#888888").rotate(a, about_point=ORIGIN)
            for a in np.linspace(0, 2*PI, 8, endpoint=False)
        ])
        blur_group = VGroup(blur_coin, spin_indicators)
        self.place_at_grid(blur_group, "D5", scale_factor=0.8)

        # Animation of merging
        self.play(
            heads_group.animate.move_to(self.grid["D5"]).set_opacity(0),
            tails_group.animate.move_to(self.grid["D5"]).set_opacity(0),
            FadeIn(blur_group),
            run_time=2
        )
        
        # Rotation updater for spin
        blur_group.add_updater(lambda m, dt: m.rotate(2 * dt))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )

        # Superposition label at C5 (Fix: moved from B5 per Issue 39)
        superposition_label = Text("Superposition", font_size=20, color="#FFFFFF")
        self.place_at_grid(superposition_label, "C5")

        self.play(Write(superposition_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight lecture line 4
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(WHITE)
        )

        # Divider between col 3 and 4 (Classical/Quantum boundary)
        mid_x = (self.grid["A3"][0] + self.grid["A4"][0]) / 2
        divider_line = Line(
            start=np.array([mid_x, 3.5, 0]),
            end=np.array([mid_x, -3.5, 0]),
            color="#555555",
            stroke_width=2
        )
        
        classical_header = Text("Classical", font_size=18, color="#555555")
        quantum_header = Text("Quantum", font_size=18, color="#FFFFFF")
        self.place_at_grid(classical_header, "A2")
        self.place_at_grid(quantum_header, "A5")

        self.play(
            Create(divider_line),
            FadeIn(classical_header),
            FadeIn(quantum_header)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight lecture line 5
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE)
        )

        # Pulse glow for the Quantum side
        quantum_glow = Circle(radius=0.8, color="#FFFFFF", stroke_width=0, fill_opacity=0.1)
        self.place_at_grid(quantum_glow, "D5")
        
        glow_tracker = ValueTracker(0.1)
        quantum_glow.add_updater(lambda m: m.set_fill(opacity=glow_tracker.get_value()))
        
        self.play(
            FadeIn(quantum_glow),
            glow_tracker.animate.set_value(0.4),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(2)

        # Cleanup updaters
        blur_group.clear_updaters()
        quantum_glow.clear_updaters()
