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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the scene layout with updated script lines
        self.setup_layout(
            "The Problem: The Noisy Channel", 
            [
                "Digital data often travels through noisy, imperfect environments.",
                "Alice sends a binary message '1011' to Bob.",
                "Random interference can flip a bit during transmission.",
                "Bob receives '1001' and detects something is wrong.",
                "We must fix this error without requesting a re-send."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Alice (Asset)
        alice_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/alice.svg").set_color(BLUE)
        alice_label = Text("Alice (Cat)", font_size=18).next_to(alice_svg, DOWN, buff=0.1)
        alice = VGroup(alice_svg, alice_label)
        self.place_in_area(alice, "A1", "B2", scale_factor=0.8)
        
        # Bob (Circle representation)
        bob_circle = Circle(radius=0.5, color=GREEN)
        bob_label = Text("Bob (Dog)", font_size=18).next_to(bob_circle, DOWN, buff=0.1)
        bob = VGroup(bob_circle, bob_label)
        # Issue 29: Bob moved to D5-E6 to avoid overlap with goal text at F
        self.place_in_area(bob, "D5", "E6", scale_factor=0.8)
        
        # Binary message '1011' in a box
        bit_vals = ["1", "0", "1", "1"]
        bits = VGroup(*[Text(b, font_size=32, color=WHITE) for b in bit_vals]).arrange(RIGHT, buff=0.2)
        msg_box = SurroundingRectangle(bits, buff=0.2, color=WHITE)
        message = VGroup(bits, msg_box)
        # Issue 28: Message starts at B3 to avoid overlap with Alice
        self.place_at_grid(message, "B3", scale_factor=0.7)
        
        self.play(Create(alice), Create(bob))
        self.play(Create(message))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Message starts moving towards Bob (pauses midway for interference)
        self.play(message.animate.move_to(self.grid["C3"]), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Noise cloud (Asset)
        cloud_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cloud.svg").set_color(YELLOW_A)
        cloud_label = Text("NOISE", font_size=16, color=BLACK).move_to(cloud_svg.get_center())
        cloud = VGroup(cloud_svg, cloud_label)
        self.place_in_area(cloud, "C3", "D4", scale_factor=0.8)
        
        # Flash noise cloud YELLOW (#FFFF00)
        self.play(FadeIn(cloud))
        self.play(cloud.animate.set_color(YELLOW), run_time=0.2, rate_func=there_and_back)
        
        # Bit 3 flips from 1 to 0 (#FF0000)
        # bits[2] is the 3rd bit in '1011'
        flipped_bit_val = Text("0", font_size=32, color=RED).move_to(bits[2])
        self.play(
            bits[2].animate.set_color(RED),
            Transform(bits[2], flipped_bit_val)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Message continues to Bob
        self.play(message.animate.move_to(self.grid["D5"]), FadeOut(cloud), run_time=1.5)
        
        # Red question mark (#FF0000) appears
        # Issue 30: Place question mark at C5 to avoid clutter with Bob
        question_mark = Text("?", font_size=48, color=RED)
        self.place_at_grid(question_mark, "C5", scale_factor=1.0)
        
        self.play(Write(question_mark))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Display Goal text
        goal_text = Text("Goal: Fix the error without re-sending.", font_size=24, color=WHITE)
        # Spans bottom row to be clear of other elements
        self.place_in_area(goal_text, "F1", "F6", scale_factor=1.0)
        
        self.play(FadeIn(goal_text))
        self.wait(2)
