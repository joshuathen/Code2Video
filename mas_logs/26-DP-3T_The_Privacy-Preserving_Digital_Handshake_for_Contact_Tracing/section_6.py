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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Phase 4: Local Risk Assessment", 
            [
                "Bob's phone downloads the list of infected Secret Keys.", 
                "It re-calculates the nicknames those keys would have used.", 
                "The phone checks these against its own stored history.", 
                "If a match is found, Bob is notified privately.", 
                "All matching happens locally on Bob's own device."
            ]
        )
        
        # Colors
        PHONE_COLOR = "#87CEEB"
        INFECTED_KEY_COLOR = "#FF0000"
        SK_COLOR = "#00FF00"
        BOX_COLOR = "#696969"
        EPHID_COLOR = "#FFFFFF"
        LOCAL_LOG_COLOR = "#00FF00"
        NOTIF_COLOR = "#FF0000"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(INFECTED_KEY_COLOR))
        
        # Cloud Icon
        cloud = VGroup(
            Circle(radius=0.3, fill_opacity=1, color=WHITE).shift(LEFT*0.2),
            Circle(radius=0.4, fill_opacity=1, color=WHITE).shift(RIGHT*0.2),
            Circle(radius=0.3, fill_opacity=1, color=WHITE).shift(UP*0.2)
        )
        self.place_at_grid(cloud, "A5", scale_factor=0.6)
        cloud_label = Text("Public Cloud", font_size=16).next_to(cloud, UP, buff=0.1)
        
        # Bob's Phone
        phone_body = RoundedRectangle(height=3, width=2, corner_radius=0.2, color=PHONE_COLOR, fill_opacity=0.1)
        self.place_in_area(phone_body, "D3", "F5", scale_factor=1.0)
        phone_label = Text("Bob's Phone", font_size=16, color=PHONE_COLOR).next_to(phone_body, DOWN, buff=0.1)
        
        # Infected Keys List
        keys_list = VGroup(*[
            Rectangle(height=0.2, width=0.8, color=INFECTED_KEY_COLOR, fill_opacity=0.8)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.1)
        self.place_at_grid(keys_list, "A5", scale_factor=0.8)
        
        self.add(cloud, cloud_label, phone_body, phone_label)
        self.play(FadeIn(keys_list))
        self.play(keys_list.animate.move_to(phone_body.get_center()), run_time=1.5)
        self.play(FadeOut(keys_list), FadeOut(cloud), FadeOut(cloud_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(EPHID_COLOR))
        
        # Downloaded SK inside phone
        sk_rect = Rectangle(height=0.3, width=1.0, color=SK_COLOR, fill_opacity=0.5)
        sk_text = Text("Infected SK", font_size=14).move_to(sk_rect)
        sk_group = VGroup(sk_rect, sk_text)
        self.place_at_grid(sk_group, "D4", scale_factor=1.0)
        
        # Derivation Box
        derivation_box = Square(side_length=1.0, color=BOX_COLOR, fill_opacity=0.8)
        derivation_text = Text("Hash\nDerivation", font_size=12).move_to(derivation_box)
        derivation_group = VGroup(derivation_box, derivation_text)
        self.place_at_grid(derivation_group, "E4", scale_factor=1.0)
        
        # Output EphIDs
        possible_ephids = VGroup(*[
            Text(f"EphID_{i}", font_size=14, color=EPHID_COLOR)
            for i in ["X", "Y", "Z"]
        ]).arrange(DOWN, buff=0.2)
        self.place_at_grid(possible_ephids, "F4", scale_factor=1.0)
        
        self.play(FadeIn(sk_group))
        self.play(sk_group.animate.move_to(derivation_box.get_center()))
        self.play(FadeIn(derivation_group))
        self.play(ReplacementTransform(sk_group, possible_ephids))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(LOCAL_LOG_COLOR))
        
        # Move Possible EphIDs to the left of the phone comparison area
        # Create Local Interaction Log
        local_log_title = Text("Local Log", font_size=16, color=LOCAL_LOG_COLOR)
        local_log_entries = VGroup(*[
            Text(f"EphID_{i}", font_size=14, color=LOCAL_LOG_COLOR)
            for i in ["P", "Q", "Y", "R"]
        ]).arrange(DOWN, buff=0.2)
        local_log_group = VGroup(local_log_title, local_log_entries).arrange(DOWN, buff=0.2)
        
        self.place_at_grid(possible_ephids, "E3", scale_factor=1.0)
        self.place_at_grid(local_log_group, "E5", scale_factor=1.0)
        
        possible_ephid_title = Text("Computed", font_size=16, color=EPHID_COLOR).next_to(possible_ephids, UP)
        
        self.play(
            FadeIn(possible_ephid_title),
            FadeIn(local_log_group),
            FadeOut(derivation_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(NOTIF_COLOR))
        
        # Match "EphID_Y"
        match_line = Line(
            possible_ephids[1].get_right(), 
            local_log_entries[2].get_left(), 
            color=NOTIF_COLOR,
            stroke_width=2
        )
        
        notif_rect = RoundedRectangle(height=0.6, width=1.5, corner_radius=0.1, color=NOTIF_COLOR, fill_opacity=0.9)
        notif_text = Text("EXPOSURE ALERT", font_size=12, color=WHITE).move_to(notif_rect)
        notif_group = VGroup(notif_rect, notif_text)
        self.place_at_grid(notif_group, "D4", scale_factor=1.2)
        
        self.play(Create(match_line))
        self.play(possible_ephids[1].animate.set_color(NOTIF_COLOR), local_log_entries[2].animate.set_color(NOTIF_COLOR))
        self.play(FadeIn(notif_group, scale=0.5))
        self.play(Indicate(notif_group))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Enclose everything in the phone area with a highlight
        highlight_rect = SurroundingRectangle(VGroup(phone_body, phone_label), color=WHITE, buff=0.2)
        local_only_text = Text("STAYS ON DEVICE", font_size=18, color=WHITE).next_to(highlight_rect, UP)
        
        self.play(Create(highlight_rect))
        self.play(Write(local_only_text))
        self.wait(2)
