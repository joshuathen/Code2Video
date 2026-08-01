import os
import numpy as np
from manim import *

# Fix the FileExistsError by ensuring the directory exists before Manim's Text mobject attempts to create it.
os.makedirs(os.path.join("media", "texts"), exist_ok=True)

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

class Section2Scene(TeachingScene):
    def construct(self):
        title_text = "Generating Digital Disguises (Ephemeral IDs)"
        lecture_lines = [
            "Each phone generates a daily Secret Key (SK).",
            "A cryptographic hash function transforms this Secret Key.",
            "This creates thousands of temporary Ephemeral IDs.",
            "IDs rotate every fifteen minutes to prevent tracking.",
            "To others, the user appears as a new person."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_SK = "#5555FF"
        COLOR_HASH = "#888888"
        COLOR_EPHID = "#FFFF55"
        COLOR_CLOCK = "#FFFF55"
        COLOR_CYAN = "#00FFFF"
        COLOR_BYSTANDER = "#CCCCCC"

        # === Animation for Lecture Line 1 ===
        # Each phone generates a daily Secret Key (SK).
        self.lecture[0].set_color(COLOR_SK)
        
        # Asset: phone icon
        phone_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg")
        phone_icon.set_color(COLOR_SK)
        self.place_at_grid(phone_icon, "B2", scale_factor=0.6)
        
        sk_text = Text("Secret Key (SK)", color=COLOR_SK, weight=BOLD)
        # Fix Issue 27: Adjust positioning of sk_text to avoid overlap
        self.place_in_area(sk_text, 'B3', 'B4', scale_factor=0.7)
        
        self.play(FadeIn(phone_icon), Write(sk_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A cryptographic hash function transforms this Secret Key.
        self.lecture[1].set_color(COLOR_HASH)
        
        # Gear icon representation
        gear_center = Circle(radius=0.4, color=COLOR_HASH, fill_opacity=0.3)
        teeth = VGroup(*[
            Rectangle(width=0.15, height=0.2, color=COLOR_HASH, fill_opacity=1).move_to(
                gear_center.get_center() + 0.45 * np.array([np.cos(a), np.sin(a), 0])
            ).rotate(a)
            for a in np.linspace(0, 2*PI, 9, endpoint=False)
        ])
        gear_icon = VGroup(gear_center, teeth)
        self.place_at_grid(gear_icon, "D3", scale_factor=0.7)
        
        hash_label = Text("Hash Function", color=COLOR_HASH)
        # Fix Issue 28: Adjust positioning of hash_label to avoid congestion
        self.place_in_area(hash_label, 'E2', 'E4', scale_factor=0.7)
        
        # Arrow from SK area to Hash area
        arrow = Arrow(start=self.grid["B3"], end=self.grid["D3"], color=COLOR_HASH, buff=0.6)
        
        self.play(FadeIn(gear_icon), Write(hash_label), Create(arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This creates thousands of temporary Ephemeral IDs.
        self.lecture[2].set_color(COLOR_EPHID)
        
        # Ephemeral IDs sequence
        ephid_1 = Text("EphID_1", color=COLOR_EPHID).scale(0.5).move_to(self.grid["D3"])
        ephid_2 = Text("EphID_2", color=COLOR_EPHID).scale(0.5).move_to(self.grid["D3"])
        ephid_3 = Text("EphID_3", color=COLOR_EPHID).scale(0.5).move_to(self.grid["D3"])
        
        target_pos = self.grid["D5"]
        
        self.play(ephid_1.animate.move_to(target_pos), run_time=1)
        self.play(FadeOut(ephid_1, shift=DOWN), ephid_2.animate.move_to(target_pos), run_time=1)
        self.play(FadeOut(ephid_2, shift=DOWN), ephid_3.animate.move_to(target_pos), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # IDs rotate every fifteen minutes to prevent tracking.
        self.lecture[3].set_color(COLOR_CLOCK)
        
        # Clock icon
        clock_face = Circle(radius=0.4, color=COLOR_CLOCK)
        clock_hand = Line(clock_face.get_center(), clock_face.get_center() + 0.3 * UP, color=COLOR_CLOCK)
        clock_icon = VGroup(clock_face, clock_hand)
        # Fix Issue 29: Move clock to B5 to decompress column 5
        self.place_at_grid(clock_icon, 'B5', scale_factor=0.7)
        
        self.play(FadeIn(clock_icon))
        
        # Rotate clock and change active ID color
        self.play(
            Rotate(clock_hand, angle=-2*PI, about_point=clock_face.get_center()),
            ephid_3.animate.set_color(COLOR_CYAN),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # To others, the user appears as a new person.
        self.lecture[4].set_color(COLOR_BYSTANDER)
        
        # Bystander icon
        b_head = Circle(radius=0.15, color=COLOR_BYSTANDER)
        b_body = Line(DOWN*0.15, DOWN*0.6, color=COLOR_BYSTANDER)
        b_arms = Line(LEFT*0.3, RIGHT*0.3, color=COLOR_BYSTANDER).shift(DOWN*0.3)
        b_legs = VGroup(
            Line(ORIGIN, DOWN*0.4 + LEFT*0.2),
            Line(ORIGIN, DOWN*0.4 + RIGHT*0.2)
        ).shift(DOWN*0.6)
        bystander_icon = VGroup(b_head, b_body, b_arms, b_legs)
        self.place_at_grid(bystander_icon, "F5", scale_factor=0.8)
        
        # Thought bubble with question mark
        bubble = Ellipse(width=1.2, height=0.7, color=COLOR_BYSTANDER, fill_opacity=0.1)
        self.place_at_grid(bubble, "E5", scale_factor=0.8)
        q_mark = Text("?", color=COLOR_BYSTANDER).scale(0.8).move_to(bubble.get_center())
        bubble_group = VGroup(bubble, q_mark)
        
        self.play(FadeIn(bystander_icon), FadeIn(bubble_group))
        self.wait(2)
