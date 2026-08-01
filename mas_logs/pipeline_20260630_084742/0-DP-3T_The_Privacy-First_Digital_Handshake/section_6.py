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
        # Setup title and lines from the target script
        title = "Local Matching: The Private Detective"
        lines = [
            "Leo’s phone periodically downloads the latest sick keys.",
            "It uses these keys to recreate potential encounter codes.",
            "The device scans its local diary for any matches.",
            "Finding a match confirms a close encounter with infection.",
            "Leo receives a private exposure alert on his phone."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_KEYS = "#FFA500"
        COLOR_RPI = "#00FFFF"
        COLOR_SCANNER = "#FFFFFF"
        COLOR_MATCH = "#FFFF00"
        COLOR_ALERT = "#FF0000"

        # Assets
        PHONE_PATH = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        # Leo's phone [Asset: ...phone.svg] downloads the 'Secret Day Keys' (#FFA500) from the Public Bulletin Board.
        
        board_rect = Rectangle(height=1.8, width=3.8, color=WHITE, stroke_width=2)
        board_title = Text("Bulletin Board", font_size=20, color=WHITE).next_to(board_rect, UP, buff=0.1)
        board = VGroup(board_rect, board_title)
        # Resolved Issue 47: Moved board to B2-C5 area
        self.place_in_area(board, "B2", "C5", scale_factor=0.8)
        
        phone_asset = SVGMobject(PHONE_PATH, color=WHITE)
        phone_title = Text("Leo's Phone", font_size=18, color=WHITE).next_to(phone_asset, DOWN, buff=0.1)
        phone = VGroup(phone_asset, phone_title)
        # Resolved Issue 49: Scale set to 0.8
        self.place_at_grid(phone, "E2", scale_factor=0.8)
        
        key_icon = RoundedRectangle(corner_radius=0.1, height=0.4, width=0.8, color=COLOR_KEYS, fill_opacity=0.8)
        key_txt = Text("Sick Key", font_size=14, color=BLACK).move_to(key_icon)
        key = VGroup(key_icon, key_txt)
        self.place_at_grid(key, "B3", scale_factor=0.8)
        
        self.play(Create(board), Create(phone))
        self.play(self.lecture[0].animate.set_color(COLOR_KEYS))
        self.play(FadeIn(key))
        self.play(key.animate.move_to(phone_asset.get_center()))
        self.play(FadeOut(key))

        # === Animation for Lecture Line 2 ===
        # Leo's phone [Asset: ...phone.svg] processes the key to recreate the code 'A12B' (#00FFFF).
        
        rpi_code = Text("A12B", font_size=24, color=COLOR_RPI)
        # Resolved Issue 48: Scale factor updated to 0.8
        self.place_at_grid(rpi_code, "D2", scale_factor=0.8)
        
        self.play(self.lecture[1].animate.set_color(COLOR_RPI))
        self.play(Write(rpi_code))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A magnifying glass (#FFFFFF) scans the 'Local Diary' and finds the matching 'A12B'.
        
        diary_rect = Rectangle(height=3.5, width=2.5, color=WHITE, stroke_width=2)
        diary_lbl = Text("Local Diary", font_size=20, color=WHITE).next_to(diary_rect, UP, buff=0.1)
        
        # Diary entries - one must match A12B
        entries = ["X72Q", "A12B", "L99P", "Z44S"]
        entry_objs = VGroup(*[Text(e, font_size=22, color=WHITE) for e in entries]).arrange(DOWN, buff=0.3)
        entry_objs.move_to(diary_rect.get_center())
        diary = VGroup(diary_rect, diary_lbl, entry_objs)
        self.place_in_area(diary, "D4", "F6", scale_factor=0.8)
        
        # Magnifying glass parts
        glass_circle = Circle(radius=0.4, color=COLOR_SCANNER, stroke_width=4)
        glass_handle = Line(glass_circle.get_bottom(), glass_circle.get_bottom() + [0.3, -0.3, 0], color=COLOR_SCANNER, stroke_width=4)
        magnifying_glass = VGroup(glass_circle, glass_handle)
        magnifying_glass.move_to(entry_objs[0].get_center())
        
        self.play(self.lecture[2].animate.set_color(COLOR_SCANNER))
        self.play(Create(diary))
        self.play(FadeIn(magnifying_glass))
        
        # Scan to the second entry (A12B)
        self.play(magnifying_glass.animate.move_to(entry_objs[1].get_center()), run_time=1.5)

        # === Animation for Lecture Line 4 ===
        # The matching code in the diary flashes yellow (#FFFF00) to indicate a hit.
        
        match_entry = entry_objs[1]
        self.play(self.lecture[3].animate.set_color(COLOR_MATCH))
        self.play(
            match_entry.animate.set_color(COLOR_MATCH),
            Flash(match_entry, color=COLOR_MATCH, line_length=0.2)
        )
        self.play(Indicate(match_entry, color=COLOR_MATCH, scale_factor=1.2))

        # === Animation for Lecture Line 5 ===
        # A red 'Exposure Warning' banner (#FF0000) appears on Leo's phone [Asset: ...phone.svg] screen.
        
        alert_box = Rectangle(height=0.4, width=1.0, fill_color=COLOR_ALERT, fill_opacity=1, stroke_width=0)
        alert_text = Text("ALERT", font_size=14, color=WHITE).move_to(alert_box)
        exposure_banner = VGroup(alert_box, alert_text).move_to(phone_asset.get_center())
        
        self.play(self.lecture[4].animate.set_color(COLOR_ALERT))
        self.play(FadeIn(exposure_banner))
        self.play(Wiggle(exposure_banner))
        
        self.wait(2)
