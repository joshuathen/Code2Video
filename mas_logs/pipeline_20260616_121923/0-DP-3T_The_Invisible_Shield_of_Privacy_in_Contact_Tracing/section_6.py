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
        # Setup layout
        lines = [
            'Bob’s phone downloads the list of infected secret keys.',
            'It locally reconstructs the IDs Alice would have broadcasted.',
            "It compares these IDs against Bob's local encounter log.",
            "A match triggers an exposure alert on Bob's phone.",
            'The central server never knows that Bob was exposed.'
        ]
        self.setup_layout("Step 4: Local Matching (The Privacy Win)", lines)

        # Assets - Replaced missing SVGMobject with procedural shapes
        phone_body = RoundedRectangle(corner_radius=0.1, height=1.8, width=1.0, color=WHITE)
        phone_screen = Rectangle(height=1.4, width=0.85, color=WHITE).shift(UP * 0.1)
        phone_button = Circle(radius=0.06, color=WHITE).shift(DOWN * 0.75)
        phone_icon = VGroup(phone_body, phone_screen, phone_button)
        
        phone_icon.set_color(WHITE)
        phone_label = Text("Bob's Phone", font_size=18).next_to(phone_icon, UP, buff=0.1)
        bob_phone = VGroup(phone_icon, phone_label)
        
        # Issue 42: Move phone lower to avoid overlap
        self.place_in_area(bob_phone, "C1", "F3", scale_factor=1.0)
        
        server_box = Rectangle(height=1.2, width=2.2, color=BLUE_B)
        server_text = Text("Health Authority\n(Infected Keys List)", font_size=14, color=BLUE_B)
        server = VGroup(server_box, server_text)
        self.place_at_grid(server, "A5", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Bob's phone takes the downloaded SK_t (#FFFF00)
        sk_keys = VGroup(*[
            Text("SK_t", color="#FFFF00", font_size=24) for _ in range(3)
        ]).arrange(RIGHT, buff=0.2)
        self.place_at_grid(sk_keys, "A5", scale_factor=1.0)
        
        self.play(FadeIn(server), FadeIn(bob_phone))
        # Moving to the upper part of the phone area to avoid Bob's phone label
        self.play(sk_keys.animate.move_to(self.grid["C2"]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Calculation box inside phone - Issue 43: Center at D2
        calc_box = Rectangle(height=1.5, width=2.0, color="#FFFF00", stroke_width=2)
        calc_text = Text("Generating\nIDs...", font_size=16, color="#FFFF00")
        calc_group = VGroup(calc_box, calc_text)
        self.place_at_grid(calc_group, "D2", scale_factor=0.9)
        
        reconstructed_ids = VGroup(
            Text("A1B2", font_size=20, color="#FFFF00"),
            Text("X9Z0", font_size=20, color=WHITE),
            Text("Q5R4", font_size=20, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        recon_label = Text("Possible Alice IDs:", font_size=14, color=WHITE)
        recon_list = VGroup(recon_label, reconstructed_ids).arrange(DOWN)
        # Issue 43: Center at D2
        self.place_at_grid(recon_list, "D2", scale_factor=1.0)

        self.play(FadeOut(sk_keys), FadeIn(calc_group))
        self.play(ReplacementTransform(calc_group, recon_list))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#8B4513") # Brown
        
        # The brown 'Local Log' (#8B4513) opens
        log_box = Rectangle(height=2.5, width=2.2, color="#8B4513", fill_opacity=0.1)
        log_title = Text("Bob's Local Log", font_size=18, color="#8B4513")
        log_ids = VGroup(
            Text("K7L8", font_size=18, color=WHITE),
            Text("A1B2", font_size=18, color=WHITE), # This will match
            Text("M3N4", font_size=18, color=WHITE)
        ).arrange(DOWN, buff=0.2)
        local_log = VGroup(log_title, log_ids).arrange(DOWN, buff=0.2)
        # Placing log to align with D2 horizontally
        self.place_in_area(local_log, "C4", "E6", scale_factor=1.0)
        self.place_in_area(log_box, "C4", "E6", scale_factor=1.0)
        
        self.play(Create(log_box), FadeIn(local_log))
        
        # A match is confirmed between the log and the calculated list.
        self.wait(0.5)
        # reconstructed_ids[0] is the "A1B2" Text object
        match_id_recon = reconstructed_ids[0]
        match_id_log = log_ids[1]
        
        indicator = DoubleArrow(match_id_recon.get_right(), match_id_log.get_left(), color=YELLOW, buff=0.1)
        
        self.play(Create(indicator))
        self.play(
            match_id_recon.animate.set_color(YELLOW),
            match_id_log.animate.set_color(YELLOW)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        # A red 'Exposure Alert' (#FF0000) flashes on Bob's phone
        alert_box = RoundedRectangle(corner_radius=0.1, height=1.0, width=2.0, color=RED, fill_opacity=0.8)
        alert_text = Text("EXPOSURE ALERT", font_size=16, color=WHITE, weight=BOLD)
        alert = VGroup(alert_box, alert_text)
        # Issue 44: Place alert relative to phone's lower position
        self.place_at_grid(alert, "F2", scale_factor=1.0)
        
        self.play(FadeOut(indicator))
        for _ in range(3):
            self.play(FadeIn(alert), run_time=0.3)
            self.play(FadeOut(alert), run_time=0.3)
        self.play(FadeIn(alert))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BLUE_B)
        
        # Visualizing server ignorance
        no_info = Cross(server_box, stroke_color=RED, stroke_width=8)
        self.play(Create(no_info))
        self.wait(2)
