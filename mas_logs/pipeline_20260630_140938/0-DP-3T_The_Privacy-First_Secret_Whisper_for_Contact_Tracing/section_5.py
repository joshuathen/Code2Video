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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "If Alice tests positive, she uploads her daily keys.",
            "These keys are shared on a public bulletin board server.",
            "Bob's phone downloads new keys and regenerates possible RPIs.",
            "Bob's phone checks for matches in its local diary.",
            "The system alerts Bob privately without notifying the government."
        ]
        self.setup_layout("Step 3: Notification & Decentralized Matching", lecture_lines)
        
        # Asset path
        phone_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        # Alice's blue phone (#0000FF) sends a silver 'Daily Key' (#C0C0C0) up to a white Cloud icon (#FFFFFF).
        self.lecture[0].set_color(BLUE)
        
        alice_phone = SVGMobject(phone_asset, color=BLUE)
        self.place_at_grid(alice_phone, "D2", scale_factor=0.8)
        
        # Cloud representation
        cloud = VGroup(
            Circle(radius=0.3, color=WHITE, fill_opacity=1),
            Circle(radius=0.4, color=WHITE, fill_opacity=1).shift(RIGHT*0.3),
            Circle(radius=0.3, color=WHITE, fill_opacity=1).shift(RIGHT*0.6),
            Circle(radius=0.3, color=WHITE, fill_opacity=1).shift(UP*0.2 + RIGHT*0.3)
        )
        self.place_at_grid(cloud, "B2", scale_factor=0.6)
        
        key = Square(side_length=0.2, color="#C0C0C0", fill_opacity=1)
        self.place_at_grid(key, "D2")
        
        self.play(FadeIn(alice_phone), FadeIn(cloud))
        self.play(key.animate.move_to(self.grid["B2"]), run_time=1.5)
        self.play(FadeOut(key))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The Cloud transforms into a white grid (bulletin board) showing several silver keys.
        self.lecture[1].set_color(WHITE)
        
        # Bulletin board area: B2 to B6
        board = Rectangle(height=0.8, width=4.5, color=WHITE)
        self.place_in_area(board, "B2", "B6")
        
        keys = VGroup(*[
            Square(side_length=0.15, color="#C0C0C0", fill_opacity=1)
            for _ in range(5)
        ])
        for i, k in enumerate(keys):
            self.place_at_grid(k, f"B{2+i}")

        self.play(ReplacementTransform(cloud, board), FadeIn(keys))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Bob's green phone (#00FF00) pulls a silver key from the board and generates white RPI dots (#FFFFFF) locally.
        self.lecture[2].set_color(GREEN)
        
        bob_phone = SVGMobject(phone_asset, color=GREEN)
        self.place_at_grid(bob_phone, "D6", scale_factor=0.8)
        
        pulled_key = keys[-1].copy()
        
        self.play(FadeIn(bob_phone))
        self.play(pulled_key.animate.move_to(self.grid["D6"]), run_time=1.5)
        
        rpi_dots = VGroup(*[
            Dot(color=WHITE, radius=0.05)
            for _ in range(4)
        ]).arrange_in_grid(2, 2, buff=0.1)
        self.place_at_grid(rpi_dots, "D6", scale_factor=1.0)
        
        self.play(FadeOut(pulled_key), FadeIn(rpi_dots))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Bob's phone compares the new white RPI dots with its Diary; one pair flashes bright green (#00FF00).
        self.lecture[3].set_color(GREEN)
        
        diary_dots = VGroup(*[
            Dot(color=WHITE, radius=0.05)
            for _ in range(4)
        ]).arrange_in_grid(2, 2, buff=0.1).shift(DOWN * 0.1)
        self.place_at_grid(diary_dots, "D6", scale_factor=1.0)
        
        matching_dot = rpi_dots[2]
        
        self.play(FadeIn(diary_dots))
        self.play(Flash(matching_dot, color=GREEN, flash_radius=0.2))
        self.play(matching_dot.animate.set_color(GREEN), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A red exclamation mark (#FF0000) appears on Bob's phone; no signal is sent back to the Cloud.
        self.lecture[4].set_color(RED)
        
        alert = Text("!", font_size=48, color=RED, weight=BOLD)
        self.place_at_grid(alert, "D6", scale_factor=0.8)
        alert.shift(UP * 0.3)
        
        self.play(Write(alert))
        self.play(Indicate(alert))
        self.wait(2)
