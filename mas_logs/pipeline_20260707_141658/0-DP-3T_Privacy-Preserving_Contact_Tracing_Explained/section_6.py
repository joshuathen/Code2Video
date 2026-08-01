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
        # Setup title and lecture lines
        title_text = "Step 4: Reporting and Notification"
        lecture_lines = [
            "Infected users upload only their secret daily keys.",
            "The central server hosts these keys for everyone.",
            "Other phones download these keys to check exposure.",
            "Devices locally reconstruct the IDs to find matches.",
            "If a match exists, the user receives a notification."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define Colors
        KEY_COLOR = "#00FFFF"
        SERVER_COLOR = "#F5F5F5"
        RECONSTRUCT_COLOR = "#ADD8E6"
        MATCH_COLOR = "#FF0000"
        
        # Define Assets (using shapes)
        alice_phone = VGroup(
            RoundedRectangle(corner_radius=0.1, height=1.5, width=0.8, color=BLUE),
            Text("Alice", font_size=16, color=BLUE).shift(UP * 0.9)
        )
        
        server = VGroup(
            Rectangle(height=1.4, width=2.0, color=SERVER_COLOR, fill_opacity=0.1),
            Text("Bulletin Board", font_size=14, color=SERVER_COLOR).shift(UP * 0.5),
            Square(side_length=0.4, color=SERVER_COLOR).shift(DOWN * 0.2)
        )
        
        bob_phone = VGroup(
            RoundedRectangle(corner_radius=0.1, height=1.5, width=0.8, color=GREEN),
            Text("Bob", font_size=16, color=GREEN).shift(UP * 0.9)
        )
        
        secret_day_key = VGroup(
            Rectangle(height=0.3, width=0.6, color=KEY_COLOR, fill_opacity=0.8),
            Text("Key", font_size=10, color=BLACK)
        )

        # === Animation for Lecture Line 1 ===
        # Line 1: Infected users upload only their secret daily keys.
        self.lecture[0].set_color(KEY_COLOR)
        self.place_at_grid(alice_phone, "B1")
        # Issue 43: server needs broader horizontal area
        self.place_in_area(server, "B4", "B5", scale_factor=1.0)
        
        upload_key = secret_day_key.copy().move_to(alice_phone.get_center())
        
        self.play(FadeIn(alice_phone), FadeIn(server))
        self.play(FadeIn(upload_key))
        self.play(upload_key.animate.move_to(server.get_center()), run_time=1.2)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Line 2: The central server hosts these keys for everyone.
        self.lecture[1].set_color(SERVER_COLOR)
        self.play(upload_key.animate.scale(0.8).shift(DOWN * 0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Other phones download these keys to check exposure.
        self.lecture[2].set_color(KEY_COLOR)
        self.place_at_grid(bob_phone, "E1")
        
        download_key = upload_key.copy()
        self.play(FadeIn(bob_phone))
        self.play(download_key.animate.move_to(bob_phone.get_center()), run_time=1.2)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Line 4: Devices locally reconstruct the IDs to find matches.
        self.lecture[3].set_color(RECONSTRUCT_COLOR)
        
        ephids = VGroup(*[
            Text(f"EphID_{i}", font_size=12, color=RECONSTRUCT_COLOR) 
            for i in range(4)
        ]).arrange(DOWN, buff=0.2)
        # Issue 42: ephids vertical crowding
        self.place_in_area(ephids, "C3", "D3", scale_factor=0.8)
        
        reconstruct_lines = VGroup(*[
            Line(bob_phone.get_right(), eph.get_left(), color=RECONSTRUCT_COLOR, stroke_width=2)
            for eph in ephids
        ])
        
        self.play(Create(reconstruct_lines), FadeIn(ephids))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: If a match exists, the user receives a notification.
        self.lecture[4].set_color(MATCH_COLOR)
        
        log_bg = Rectangle(height=1.8, width=2.4, color=WHITE, fill_opacity=0.05)
        log_title = Text("Bob's Log", font_size=14, color=WHITE).shift(UP * 0.7)
        log_rows = VGroup(*[
            VGroup(Text(label, font_size=11), Text("12:00", font_size=9)).arrange(RIGHT, buff=0.3)
            for label in ["ID_7x2", "EphID_1", "ID_9b1"]
        ]).arrange(DOWN, buff=0.15)
        
        log = VGroup(log_bg, log_title, log_rows)
        self.place_in_area(log, "E4", "F6")
        
        self.play(FadeIn(log))
        self.wait(0.5)
        
        matching_ephid = ephids[1] # EphID_1
        target_row = log_rows[1]
        match_highlight = Rectangle(height=0.3, width=2.2, color=MATCH_COLOR, fill_opacity=0.4).move_to(target_row)
        
        self.play(
            matching_ephid.animate.move_to(target_row[0].get_center()).set_color(MATCH_COLOR),
            target_row[0].animate.set_color(MATCH_COLOR),
            FadeIn(match_highlight)
        )
        
        alert_box = RoundedRectangle(corner_radius=0.1, height=0.7, width=1.4, color=MATCH_COLOR, fill_opacity=0.8)
        alert_text = Text("ALERT!", font_size=14, color=WHITE)
        alert = VGroup(alert_box, alert_text)
        # Issue 41: alert overlaps alice_phone at B1
        self.place_at_grid(alert, "B1", scale_factor=0.5) 
        
        self.play(GrowFromCenter(alert))
        self.wait(2)
