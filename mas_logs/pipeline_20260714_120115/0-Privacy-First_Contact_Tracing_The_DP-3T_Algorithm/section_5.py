from manim import *
import numpy as np

# Focused fix for FileExistsError: redirecting the text cache to a new directory
config.text_dir = "media/text_cache"

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
        # Fetching data from storyboard
        title_text = "Local Matching: The Final Privacy Shield"
        lecture_lines = [
            "Phones periodically download the list of infected keys.",
            "Each phone locally regenerates the corresponding Ephemeral IDs.",
            "It checks for matches within its own seen diary.",
            "Matching occurs privately without notifying the central server.",
            "Users get alerts while maintaining total digital anonymity."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        BLUE_KEY = "#5555FF"
        GEAR_COLOR = "#888888"
        HIGHLIGHT = "#FFFF00"
        NOTIF_COLOR = "#00FFFF"
        PHONE_COLOR = "#444444"

        # Assets
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_KEY)
        
        # Bulletin Board
        board = Rectangle(width=3.5, height=1.0, color=WHITE, fill_opacity=0.1)
        self.place_in_area(board, "B2", "B5", scale_factor=0.8)
        board_label = Text("Bulletin Board", font_size=16).next_to(board, UP, buff=0.1)
        
        # Nexus Phone [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg]
        phone = SVGMobject(PHONE_ASSET, color=WHITE, fill_color=PHONE_COLOR, fill_opacity=1)
        self.place_in_area(phone, "E3", "F4", scale_factor=0.6)
        phone_label = Text("Nexus's Phone", font_size=16).next_to(phone, DOWN, buff=0.1)

        # Infected Keys
        keys = VGroup(*[
            Square(side_length=0.25, fill_color=BLUE_KEY, fill_opacity=1, stroke_width=1) 
            for _ in range(3)
        ]).arrange(RIGHT, buff=0.2)
        self.place_in_area(keys, "B3", "B4", scale_factor=0.8)

        self.add(board, board_label, phone, phone_label)
        self.play(FadeIn(keys))
        self.play(keys.animate.move_to(phone.get_center()).scale(0.4), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GEAR_COLOR)
        
        # Gear inside phone [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg]
        gear = VGroup(
            Circle(radius=0.25, color=GEAR_COLOR, stroke_width=3),
            *[Line(start=UP*0.2, end=UP*0.35, color=GEAR_COLOR, stroke_width=3).rotate(a, about_point=ORIGIN) 
              for a in np.linspace(0, 2*PI, 9)[:-1]]
        )
        gear.move_to(phone.get_center())
        
        # Set B (Regenerated IDs) - Issue 37: Move to D4, scale 0.8
        set_b_box = RoundedRectangle(width=1.8, height=1.3, color=WHITE, fill_opacity=0.05)
        self.place_at_grid(set_b_box, "D4", scale_factor=0.8)
        set_b_label = Text("Set B (Infected)", font_size=14).next_to(set_b_box, UP, buff=0.1)
        id_b_1 = Text("EphID_88", font_size=16, color=WHITE)
        id_b_2 = Text("EphID_12", font_size=16, color=WHITE)
        set_b_content = VGroup(id_b_1, id_b_2).arrange(DOWN, buff=0.15).move_to(set_b_box.get_center())

        self.play(FadeIn(gear))
        self.play(Rotate(gear, angle=2*PI, run_time=1.5))
        self.play(FadeIn(set_b_box, set_b_label, set_b_content))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREY_B)
        
        # Set A (Seen Diary) [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg]
        # Issue 36: Move to D3, scale 0.8
        set_a_box = RoundedRectangle(width=1.8, height=1.3, color=WHITE, fill_opacity=0.05)
        self.place_at_grid(set_a_box, "D3", scale_factor=0.8)
        set_a_label = Text("Set A (Seen)", font_size=14).next_to(set_a_box, UP, buff=0.1)
        id_a_1 = Text("EphID_45", font_size=16, color=WHITE)
        id_a_2 = Text("EphID_88", font_size=16, color=WHITE)
        set_a_content = VGroup(id_a_1, id_a_2).arrange(DOWN, buff=0.15).move_to(set_a_box.get_center())

        self.play(FadeIn(set_a_box, set_a_label, set_a_content))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(HIGHLIGHT)
        
        # Highlight match
        match_highlight_a = SurroundingRectangle(id_a_2, color=HIGHLIGHT, buff=0.05)
        match_highlight_b = SurroundingRectangle(id_b_1, color=HIGHLIGHT, buff=0.05)

        self.play(Create(match_highlight_a), Create(match_highlight_b))
        self.play(Indicate(id_a_2, color=HIGHLIGHT), Indicate(id_b_1, color=HIGHLIGHT))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(NOTIF_COLOR)
        
        # Notification bubble [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg]
        # Issue 38: Scale to 0.7
        notif_bubble = RoundedRectangle(width=2.5, height=0.6, corner_radius=0.1, color=NOTIF_COLOR, fill_color=NOTIF_COLOR, fill_opacity=0.2)
        self.place_in_area(notif_bubble, "C3", "C4", scale_factor=0.7)
        notif_text = Text("Exposure Detected", font_size=16, color=NOTIF_COLOR).move_to(notif_bubble.get_center())
        
        self.play(FadeIn(notif_bubble, notif_text))
        self.play(phone.animate.set_stroke(color=NOTIF_COLOR, width=6))
        self.wait(2)
