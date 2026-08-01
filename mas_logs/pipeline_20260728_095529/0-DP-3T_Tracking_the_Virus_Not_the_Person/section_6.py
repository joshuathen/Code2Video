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
        self.setup_layout("Step 4: The Local Match (Privacy Victory)", [
            "Bob’s phone periodically downloads the latest infected keys.",
            "It reconstructs the IDs those keys would have generated.",
            "It then compares these IDs against its local log.",
            "If a match exists, Bob receives a private notification."
        ])

        # Colors
        CLOUD_COLOR = "#FFFFFF"
        KEY_COLOR = "#FFD700"
        PHONE_COLOR = "#FFA500"
        TARGET_ID_COLOR = "#00FFFF"
        LOG_COLOR = "#808080"
        MATCH_COLOR = "#FF0000"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Cloud
        cloud = VGroup(
            Circle(radius=0.5, color=CLOUD_COLOR, fill_opacity=0.3),
            Circle(radius=0.35, color=CLOUD_COLOR, fill_opacity=0.3).shift(LEFT * 0.4 + DOWN * 0.1),
            Circle(radius=0.35, color=CLOUD_COLOR, fill_opacity=0.3).shift(RIGHT * 0.4 + DOWN * 0.1)
        )
        self.place_in_area(cloud, "A3", "A4", scale_factor=0.7)
        
        # Bob's Phone (Visual Container)
        # Centering the phone in the C3-F5 area
        phone_body = RoundedRectangle(height=4.2, width=3.2, corner_radius=0.2, color=PHONE_COLOR)
        phone_screen = Rectangle(height=3.8, width=3.0, color=WHITE, fill_opacity=0.1)
        phone = VGroup(phone_body, phone_screen)
        self.place_in_area(phone, "C3", "F5", scale_factor=1.0)
        
        # Infected Key descending from cloud
        sk_infected_obj = Text("SK_Infected", font_size=20, color=KEY_COLOR)
        self.place_in_area(sk_infected_obj, "A3", "A4", scale_factor=1.0)

        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.play(Create(cloud), Create(phone))
        self.play(sk_infected_obj.animate.move_to(phone.get_top() + DOWN * 0.5))
        self.play(FadeOut(sk_infected_obj))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Internal processing: SK_Infected to Target IDs
        sk_label = Text("SK_Infected", font_size=16, color=KEY_COLOR)
        # Issue 37: Position sk_label at the top center of the phone UI
        self.place_in_area(sk_label, 'C3', 'C5', scale_factor=0.8)
        
        target_ids_group = VGroup(
            Text("Target IDs:", font_size=14, color=WHITE),
            Text("ID_101", font_size=14, color=TARGET_ID_COLOR),
            Text("ID_102", font_size=14, color=TARGET_ID_COLOR),
            Text("ID_103", font_size=14, color=TARGET_ID_COLOR)
        ).arrange(DOWN, buff=0.1)
        # Issue 39: Move target_ids_group to the left side of the phone UI
        self.place_in_area(target_ids_group, 'D2', 'D4', scale_factor=0.8)
        
        proc_arrow = Arrow(sk_label.get_bottom(), target_ids_group.get_top(), buff=0.1, color=WHITE)
        
        self.play(FadeIn(sk_label))
        self.play(GrowArrow(proc_arrow), Write(target_ids_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Bob's local Encounter Log
        log_entries_group = VGroup(
            Text("Encounter Log:", font_size=14, color=WHITE),
            Text("ID_99", font_size=14, color=LOG_COLOR),
            Text("ID_102", font_size=14, color=LOG_COLOR),
            Text("ID_105", font_size=14, color=LOG_COLOR)
        ).arrange(DOWN, buff=0.1)
        # Positioned on the right side of the phone UI
        self.place_at_grid(log_entries_group, "D5", scale_factor=0.8)
        
        compare_arrow = DoubleArrow(target_ids_group.get_right(), log_entries_group.get_left(), buff=0.2, color=WHITE)
        
        self.play(Write(log_entries_group))
        self.play(Create(compare_arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Identify match (ID_102)
        match_rect_1 = SurroundingRectangle(target_ids_group[2], color=MATCH_COLOR, buff=0.05)
        match_rect_2 = SurroundingRectangle(log_entries_group[2], color=MATCH_COLOR, buff=0.05)
        
        # Private Notification popup
        notif_box = RoundedRectangle(height=0.5, width=1.4, corner_radius=0.1, color=WHITE, fill_opacity=0.9)
        notif_text = Text("MATCH FOUND", font_size=12, color=BLACK, weight=BOLD)
        notification = VGroup(notif_box, notif_text)
        # Issue 38: Center the notification at the bottom of the phone UI
        self.place_in_area(notification, 'F3', 'F5', scale_factor=0.8)
        
        self.play(
            Create(match_rect_1),
            Create(match_rect_2),
            target_ids_group[2].animate.set_color(MATCH_COLOR),
            log_entries_group[2].animate.set_color(MATCH_COLOR)
        )
        self.play(Flash(log_entries_group[2], color=MATCH_COLOR))
        self.play(FadeIn(notification))
        self.wait(2)
