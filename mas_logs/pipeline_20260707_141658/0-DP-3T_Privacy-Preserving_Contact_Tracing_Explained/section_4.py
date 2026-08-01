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

class Section4Scene(TeachingScene):
    def construct(self):
        title_text = "Step 2: The Digital Handshake (Ephemeral IDs)"
        lecture_lines = [
            "Daily keys generate multiple short-term ephemeral IDs.",
            "These IDs rotate frequently, roughly every fifteen minutes.",
            "Phones broadcast these random IDs via Bluetooth signals.",
            "Nearby devices listen and collect these rotating IDs.",
            "This handshake contains no personal or location data."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        ALICE_PHONE_COLOR = "#C0C0C0"
        BOB_PHONE_COLOR = "#ADD8E6"
        EPHID1_COLOR = "#00FF00"
        EPHID2_COLOR = "#FF00FF"
        HIGHLIGHT_COLOR = YELLOW

        # Assets
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        TIMER_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/timer.svg"

        # Alice's Phone
        alice_phone = SVGMobject(PHONE_ASSET).set_color(ALICE_PHONE_COLOR)
        self.place_at_grid(alice_phone, "C2", scale_factor=0.6)
        alice_label = Text("Alice", font_size=18, color=ALICE_PHONE_COLOR)
        alice_label.next_to(alice_phone, UP, buff=0.1)

        # Bob's Phone
        bob_phone = SVGMobject(PHONE_ASSET).set_color(BOB_PHONE_COLOR)
        self.place_at_grid(bob_phone, "C5", scale_factor=0.6)
        bob_label = Text("Bob", font_size=18, color=BOB_PHONE_COLOR)
        bob_label.next_to(bob_phone, UP, buff=0.1)

        # Timer
        timer_svg = SVGMobject(TIMER_ASSET).set_color(WHITE)
        timer_text = Text("15 min", font_size=16, color=WHITE).next_to(timer_svg, DOWN, buff=0.1)
        timer = VGroup(timer_svg, timer_text)
        self.place_at_grid(timer, "A2", scale_factor=0.6)

        # Pulse Rings
        pulse1 = Circle(radius=0.1, color=EPHID1_COLOR, stroke_opacity=0.8)
        pulse1.move_to(alice_phone.get_center())
        
        pulse2 = Circle(radius=0.1, color=EPHID2_COLOR, stroke_opacity=0.8)
        pulse2.move_to(alice_phone.get_center())

        # EphID Labels
        ephid1_text = Text("EphID_1", font_size=18, color=EPHID1_COLOR)
        ephid2_text = Text("EphID_2", font_size=18, color=EPHID2_COLOR)
        self.place_at_grid(ephid1_text, "C4", scale_factor=0.9)
        self.place_at_grid(ephid2_text, "C4", scale_factor=0.9)

        # No Data Icon
        no_data_icon = VGroup(
            Text("GPS", font_size=18, color=RED),
            Line(LEFT*0.3, RIGHT*0.3, color=RED).rotate(PI/4),
            Line(LEFT*0.3, RIGHT*0.3, color=RED).rotate(-PI/4)
        )
        self.place_at_grid(no_data_icon, "E4", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Daily keys generate multiple short-term ephemeral IDs.
        self.play(self.lecture[0].animate.set_color(EPHID1_COLOR))
        self.play(FadeIn(alice_phone), FadeIn(alice_label))
        self.play(Create(pulse1), FadeIn(ephid1_text))
        self.play(pulse1.animate.scale(8).set_stroke(opacity=0), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # These IDs rotate frequently, roughly every fifteen minutes.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        self.play(FadeIn(timer))
        self.play(Rotate(timer_svg, angle=-2*PI), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Phones broadcast these random IDs via Bluetooth signals.
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(EPHID2_COLOR))
        self.play(FadeOut(ephid1_text), FadeIn(ephid2_text))
        self.play(Create(pulse2))
        self.play(pulse2.animate.scale(8).set_stroke(opacity=0.4), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Nearby devices listen and collect these rotating IDs.
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(BOB_PHONE_COLOR))
        self.play(FadeIn(bob_phone), FadeIn(bob_label))
        # Reception animation
        self.play(pulse2.animate.scale(1.5).set_stroke(opacity=0).move_to(bob_phone.get_center()), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # This handshake contains no personal or location data.
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(RED))
        self.play(FadeIn(no_data_icon))
        self.wait(2)
        
        # Cleanup colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
