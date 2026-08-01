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
        # Initialize the layout
        lecture_lines = [
            "Phones download daily keys of diagnosed users.",
            "Your phone locally reconstructs all possible identifiers.",
            "It checks these against your stored contact diary.",
            "A match triggers a private exposure notification."
        ]
        self.setup_layout("Phase 3: The Local Match (Verification)", lecture_lines)

        # Assets
        phone_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        diary_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/diary.svg"

        # Colors
        COLOR_L1 = BLUE_B
        COLOR_L2 = YELLOW_B
        COLOR_L3 = GREEN_B
        COLOR_L4 = "#FF0000" # Exact red as requested

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_L1)
        
        # Public Bulletin Board
        board_rect = Rectangle(width=2, height=1.5, color=WHITE)
        self.place_in_area(board_rect, "A5", "B6", scale_factor=0.8)
        board_label = Text("Public Board", font_size=16).next_to(board_rect, UP, buff=0.1)
        
        # Ace's Phone
        phone = SVGMobject(phone_path, color=WHITE)
        self.place_at_grid(phone, "B2", scale_factor=0.6)
        phone_label = Text("Ace's Phone", font_size=16).next_to(phone, UP, buff=0.1)

        # Daily Key from board
        daily_key = VGroup(
            Square(side_length=0.4, color=COLOR_L1, fill_opacity=0.8),
            Text("Key", font_size=12, color=WHITE)
        )
        self.place_at_grid(daily_key, "B5", scale_factor=1.0)

        self.add(board_rect, board_label, phone, phone_label)
        self.play(FadeIn(daily_key))
        self.play(daily_key.animate.move_to(phone.get_center()), run_time=1.5)
        self.play(FadeOut(daily_key))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_L2)

        # Phone generates RPIs
        rpi_list_title = Text("Generated RPIs", font_size=16, color=COLOR_L2)
        self.place_at_grid(rpi_list_title, "C2", scale_factor=1.0)
        
        rpi_entries = VGroup(
            Text("ABC-123", font_size=18),
            Text("XYZ-789", font_size=18),
            Text("MATCH-01", font_size=18)
        ).arrange(DOWN, buff=0.2)
        self.place_at_grid(rpi_entries, "D2", scale_factor=0.8)

        self.play(FadeIn(rpi_list_title))
        self.play(Write(rpi_entries))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_L3)

        # Diary Asset
        diary = SVGMobject(diary_path, color=COLOR_L3)
        self.place_at_grid(diary, "B4", scale_factor=0.6)
        diary_label = Text("Contact Diary", font_size=16, color=COLOR_L3).next_to(diary, UP, buff=0.1)

        # Diary Contents
        diary_entries = VGroup(
            Text("JKL-456", font_size=18),
            Text("MATCH-01", font_size=18),
            Text("MNO-789", font_size=18)
        ).arrange(DOWN, buff=0.2)
        self.place_at_grid(diary_entries, "D4", scale_factor=0.8)

        self.play(FadeIn(diary), FadeIn(diary_label))
        self.play(Write(diary_entries))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_L4)

        # Highlight the match
        match_gen = rpi_entries[2]
        match_diary = diary_entries[1]

        self.play(
            match_gen.animate.set_color(COLOR_L4),
            match_diary.animate.set_color(COLOR_L4)
        )
        
        # Phone warning flash
        warning_text = Text("EXPOSURE!", font_size=24, color=COLOR_L4, weight=BOLD)
        self.place_at_grid(warning_text, "A2", scale_factor=1.0)

        self.play(
            FadeIn(warning_text),
            phone.animate.set_color(COLOR_L4)
        )
        
        # Pulse effect
        for _ in range(3):
            self.play(phone.animate.scale(1.2), run_time=0.2)
            self.play(phone.animate.scale(1/1.2), run_time=0.2)

        self.wait(2)
