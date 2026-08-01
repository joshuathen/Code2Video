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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout
        lines = [
            'If Bob tests positive, he uploads his daily key.',
            'The central server acts as a public bulletin board.',
            'Others download the keys and reconstruct potential IDs.',
            "Alice's phone matches downloaded IDs against her local diary.",
            'Matching IDs trigger a private local exposure alert.'
        ]
        self.setup_layout("Phase 3: Exposure Notification (The Bulletin Board)", lines)

        # Common Colors
        BOB_COLOR = "#e74c3c"
        KEY_COLOR = "#3498db"
        SERVER_COLOR = "#95a5a6"
        ALICE_COLOR = "#2ecc71"
        ID_COLOR = "#f1c40f"
        ALERT_COLOR = "#e74c3c"

        # === Animation for Lecture Line 1 ===
        # Bob uploads Daily Key
        self.lecture[0].set_color(BOB_COLOR)
        
        bob_phone = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.8, color=WHITE)
        bob_label = Text("Bob", font_size=16).next_to(bob_phone, UP, buff=0.1)
        bob_group = VGroup(bob_phone, bob_label)
        self.place_at_grid(bob_group, "B2") # Updated position
        
        server = Rectangle(height=1.5, width=2.5, color=SERVER_COLOR, fill_opacity=0.2)
        server_label = Text("Public Server", font_size=18).move_to(server.get_center())
        server_group = VGroup(server, server_label)
        self.place_in_area(server_group, "B4", "C6")

        daily_key = Text("Key_Bob", font_size=14, color=KEY_COLOR)
        self.place_at_grid(daily_key, "B2")

        self.play(FadeIn(bob_group), FadeIn(server_group))
        self.wait(0.5)
        
        # Bob tests positive (turns red)
        self.play(bob_phone.animate.set_fill(BOB_COLOR, opacity=0.5))
        
        # Upload key
        self.play(daily_key.animate.move_to(self.grid["B4"]), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Server as bulletin board
        self.lecture[1].set_color(SERVER_COLOR)
        
        bulletin_board_text = Text("Public Bulletin Board", font_size=14, color=SERVER_COLOR)
        self.place_at_grid(bulletin_board_text, "A5")
        
        self.play(Write(bulletin_board_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Others (Alice) download keys and reconstruct IDs
        self.lecture[2].set_color(ID_COLOR)
        
        alice_phone = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.8, color=WHITE)
        alice_label = Text("Alice", font_size=16).next_to(alice_phone, UP, buff=0.1)
        alice_group = VGroup(alice_phone, alice_label)
        self.place_at_grid(alice_group, "E2") # Updated position
        
        download_arrow = Arrow(start=self.grid["C5"], end=self.grid["E2"], color=WHITE, buff=0.5)
        
        self.play(FadeIn(alice_group))
        self.play(GrowArrow(download_arrow))
        
        downloaded_key = daily_key.copy()
        self.play(downloaded_key.animate.move_to(self.grid["E2"]), run_time=1)
        self.play(FadeOut(download_arrow))

        # Reconstruct IDs inside phone
        potential_ids_title = Text("Potential EphIDs", font_size=16, color=ID_COLOR)
        self.place_at_grid(potential_ids_title, "D4") # Updated position
        
        id_list = VGroup(
            Text("B2z8", font_size=14, color=ID_COLOR),
            Text("X9a1", font_size=14, color=ID_COLOR),
            Text("Q4p0", font_size=14, color=ID_COLOR)
        ).arrange(DOWN, buff=0.15)
        self.place_at_grid(id_list, "E4") # Updated position

        self.play(Write(potential_ids_title), FadeIn(id_list))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Match against local diary
        self.lecture[3].set_color(ALICE_COLOR)
        
        diary_title = Text("Local Diary", font_size=16, color=ALICE_COLOR)
        self.place_at_grid(diary_title, "D6") # Updated position
        
        diary_list = VGroup(
            Text("M7v2", font_size=14, color=WHITE),
            Text("B2z8", font_size=14, color=WHITE), # This is the match
            Text("K3j5", font_size=14, color=WHITE)
        ).arrange(DOWN, buff=0.15)
        self.place_at_grid(diary_list, "E6") # Updated position
        
        self.play(Write(diary_title), FadeIn(diary_list))
        self.wait(0.5)
        
        # Highlight matches
        match_highlight_1 = SurroundingRectangle(id_list[0], color=ALERT_COLOR, buff=0.05)
        match_highlight_2 = SurroundingRectangle(diary_list[1], color=ALERT_COLOR, buff=0.05)
        
        self.play(Create(match_highlight_1), Create(match_highlight_2))
        self.play(
            id_list[0].animate.set_color(ALERT_COLOR),
            diary_list[1].animate.set_color(ALERT_COLOR)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Exposure Alert
        self.lecture[4].set_color(ALERT_COLOR)
        
        alert_box = Rectangle(height=0.4, width=1.4, color=ALERT_COLOR, fill_opacity=0.9)
        alert_text = Text("EXPOSURE!", font_size=12, color=WHITE, weight=BOLD)
        alert_group = VGroup(alert_box, alert_text)
        self.place_at_grid(alert_group, "E2") # On top of Alice's phone
        
        self.play(FadeIn(alert_group))
        self.play(Flash(alert_group, color=ALERT_COLOR, flash_radius=0.5))
        self.wait(2)
