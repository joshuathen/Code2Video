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
        self.setup_layout("Step 2: The Digital Handshake", [
            "Nearby phones exchange their current anonymous IDs.",
            "Devices only store the IDs they receive from others.",
            "No names or location data are ever exchanged."
        ])

        # Colors
        ALICE_COLOR = "#5555FF"
        BOB_COLOR = "#FFA500"
        DENIED_COLOR = "#FF0000"
        
        # Asset path
        phone_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ALICE_COLOR)
        
        # Phone Alice (Asset Integration)
        phone_alice = SVGMobject(phone_asset_path, color=ALICE_COLOR)
        self.place_at_grid(phone_alice, "B2", scale_factor=0.6)
        alice_label = Text("Alice", font_size=16, color=ALICE_COLOR)
        alice_label.next_to(phone_alice, DOWN, buff=0.1)
        
        # Phone Bob (Asset Integration)
        phone_bob = SVGMobject(phone_asset_path, color=BOB_COLOR)
        self.place_at_grid(phone_bob, "B5", scale_factor=0.6)
        bob_label = Text("Bob", font_size=16, color=BOB_COLOR)
        bob_label.next_to(phone_bob, DOWN, buff=0.1)

        # IDs
        id_a = Text("ID_A", font_size=18, color=ALICE_COLOR)
        id_b = Text("ID_B", font_size=18, color=BOB_COLOR)
        self.place_at_grid(id_a, "B2")
        self.place_at_grid(id_b, "B5")

        self.play(FadeIn(phone_alice), FadeIn(alice_label), FadeIn(phone_bob), FadeIn(bob_label))
        self.wait(0.5)
        
        # Exchange animation
        self.play(
            id_a.animate.move_to(self.grid["B5"]),
            id_b.animate.move_to(self.grid["B2"]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BOB_COLOR)

        # Encounter Log Folder/Box (Fixing issues 31 & 32)
        log_label = Text("Encounter Log", font_size=18, color=WHITE)
        self.place_at_grid(log_label, "C5") # Issue 31 fix
        
        log_box = Rectangle(height=1.2, width=1.5, color=WHITE)
        self.place_in_area(log_box, "D5", "E6", scale_factor=0.85) # Issue 32 fix
        
        log_group = VGroup(log_label, log_box)
        
        self.play(Create(log_group))
        
        # Move ID_A (received by Bob) into Bob's log
        self.play(
            id_a.animate.scale(0.8).move_to(log_box.get_center()),
            FadeOut(id_b), # Bob doesn't store his own ID_B in his log for this specific visualization focus
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(DENIED_COLOR)

        # Privacy info (Fixing issue 31 & 33)
        name_text = Text("Name: Alice", font_size=20, color=WHITE)
        loc_text = Text("Location: GPS(X,Y)", font_size=20, color=WHITE)
        
        self.place_at_grid(name_text, "C2", scale_factor=0.8) # Issue 31 fix
        self.place_at_grid(loc_text, "D2", scale_factor=0.8) # Issue 33 fix
        
        self.play(Write(name_text), Write(loc_text))
        
        # Denied Crosses
        cross_1 = Cross(name_text, stroke_color=DENIED_COLOR)
        cross_2 = Cross(loc_text, stroke_color=DENIED_COLOR)
        
        self.play(Create(cross_1), Create(cross_2))
        self.wait(2)
        
        # Reset colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
