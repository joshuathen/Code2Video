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
        # Initialize the layout with updated lines from prompt
        lecture_lines = [
            'Phones exchange nicknames when in close proximity.',
            "Alice broadcasts her ephemeral ID to Bob's phone.",
            'Bob reciprocates by sending his ID to Alice.',
            'Alice records Bob’s ID in her local diary.',
            'Bob stores Alice’s ID privately on his device.'
        ]
        self.setup_layout("Phase 2: The Digital Handshake", lecture_lines)

        # Create Alice and Bob Icons
        def create_person(color, label_text):
            head = Circle(radius=0.25, color=color, fill_opacity=0.8)
            body = Polygon([-0.4, -0.6, 0], [0.4, -0.6, 0], [0, 0, 0], color=color, fill_opacity=0.8)
            phone = Rectangle(width=0.2, height=0.35, color=GREY_B, fill_opacity=1).shift(RIGHT*0.3 + DOWN*0.2)
            label = Text(label_text, font_size=18, color=WHITE).next_to(body, DOWN, buff=0.1)
            return VGroup(head, body, phone, label)

        alice = create_person(color=PINK, label_text="Alice")
        bob = create_person(color=BLUE, label_text="Bob")
        
        self.place_at_grid(alice, "B1", scale_factor=0.8)
        self.place_at_grid(bob, "B6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(alice), FadeIn(bob))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        packet_a = VGroup(
            Rectangle(width=0.8, height=0.4, color=WHITE, fill_opacity=1),
            Text("EphID_A", font_size=14, color=BLACK)
        )
        self.place_at_grid(packet_a, "B2", scale_factor=0.7)
        self.play(packet_a.animate.move_to(self.grid["B5"]), run_time=1.5)
        self.play(FadeOut(packet_a))

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        packet_b = VGroup(
            Rectangle(width=0.8, height=0.4, color=WHITE, fill_opacity=1),
            Text("EphID_B", font_size=14, color=BLACK)
        )
        self.place_at_grid(packet_b, "B5", scale_factor=0.7)
        self.play(packet_b.animate.move_to(self.grid["B2"]), run_time=1.5)
        self.play(FadeOut(packet_b))

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )

        def create_diary(title):
            box = Rectangle(width=1.8, height=2.0, color=GREY_A)
            header = Text(title, font_size=16, color=WHITE).shift(UP*0.7)
            line = Line(LEFT*0.8, RIGHT*0.8).next_to(header, DOWN, buff=0.1)
            entries = VGroup(
                Text("Contact Diary", font_size=12, color=GREY_C),
                Text("...", font_size=12, color=GREY_C)
            ).arrange(DOWN, buff=0.2).next_to(line, DOWN, buff=0.2)
            return VGroup(box, header, line, entries)

        alice_diary = create_diary("Alice's Phone")
        # Fix Issue 47 & 49: D2-E3, scale 0.8
        self.place_in_area(alice_diary, "D2", "E3", scale_factor=0.8)
        
        self.play(FadeIn(alice_diary))
        entry_b = Text("EphID_B", font_size=14, color=GREEN_B)
        entry_b.move_to(alice_diary[3].get_bottom() + DOWN*0.2)
        self.play(Write(entry_b))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        bob_diary = create_diary("Bob's Phone")
        # Fix Issue 48 & 49: D4-E5, scale 0.8
        self.place_in_area(bob_diary, "D4", "E5", scale_factor=0.8)
        
        self.play(FadeIn(bob_diary))
        entry_a = Text("EphID_A", font_size=14, color=GREEN_B)
        entry_a.move_to(bob_diary[3].get_bottom() + DOWN*0.2)
        self.play(Write(entry_a))
        
        # Final wait
        self.wait(2)
