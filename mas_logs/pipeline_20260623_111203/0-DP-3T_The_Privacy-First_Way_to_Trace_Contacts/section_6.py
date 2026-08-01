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
        # Setup
        lecture_lines = [
            "Bob’s phone downloads new positive keys from the server.",
            "It reconstructs all Ephemeral IDs from those secret keys.",
            "Bob's phone compares these against IDs in its notebook.",
            "If a match is found, Bob is notified privately.",
            "The matching process happens entirely on Bob’s device."
        ]
        self.setup_layout("Step 4: Local Matching and Notification", lecture_lines)

        # Assets & Colors
        server_color = BLUE_D
        phone_color = GREY_B
        id_color = "#3498DB"
        match_color = "#2ECC71"
        alert_color = "#E74C3C"
        local_process_color = "#F1C40F"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(server_color)
        
        server = VGroup(
            Rectangle(height=1.2, width=1.0, color=server_color, fill_opacity=0.2),
            Text("Server", font_size=18, color=server_color).shift(DOWN * 0.4)
        )
        self.place_at_grid(server, "A3")
        
        phone_body = Rectangle(height=2.2, width=1.4, color=phone_color, stroke_width=4)
        phone_label = Text("Bob's Phone", font_size=16, color=WHITE).next_to(phone_body, UP, buff=0.1)
        phone = VGroup(phone_body, phone_label)
        self.place_in_area(phone, "D3", "F4")
        
        keys = VGroup(*[Square(side_length=0.15, color=server_color, fill_opacity=1) for _ in range(3)])
        keys.arrange(RIGHT, buff=0.1)
        self.place_at_grid(keys, "A3")
        
        self.play(FadeIn(server), FadeIn(phone))
        self.play(keys.animate.move_to(self.grid["D3"]), run_time=1.5)
        self.play(FadeOut(keys, target_position=self.grid["E3"]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(id_color)
        
        recon_box = Rectangle(height=0.8, width=1.8, color=id_color, fill_opacity=0.1)
        recon_text = Text("Reconstructed IDs", font_size=14, color=id_color)
        recon_container = VGroup(recon_box, recon_text).arrange(UP, buff=0.05)
        self.place_at_grid(recon_container, "E3", scale_factor=0.8)
        
        eph_ids = VGroup(*[
            Text(f"ID_{i*123:03}", font_size=12, color=id_color) for i in range(3)
        ]).arrange(DOWN, buff=0.1)
        eph_ids.move_to(recon_box.get_center())
        
        self.play(Create(recon_box), Write(recon_text))
        self.play(FadeIn(eph_ids, shift=UP*0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(match_color)
        
        local_log_box = Rectangle(height=0.8, width=1.8, color=WHITE, fill_opacity=0.1)
        local_log_text = Text("Local Log", font_size=14, color=WHITE)
        local_log_container = VGroup(local_log_box, local_log_text).arrange(UP, buff=0.05)
        self.place_at_grid(local_log_container, "F3", scale_factor=0.8)
        
        local_ids = VGroup(
            Text("ID_999", font_size=12, color=WHITE),
            Text("ID_123", font_size=12, color=WHITE), # Match with index 1 (ID_123)
            Text("ID_456", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.1)
        local_ids.move_to(local_log_box.get_center())
        
        self.play(Create(local_log_box), Write(local_log_text))
        self.play(FadeIn(local_ids, shift=UP*0.2))
        
        # Match animation
        match_highlight_recon = eph_ids[1].copy().set_color(match_color)
        match_highlight_local = local_ids[1].copy().set_color(match_color)
        
        self.play(
            eph_ids[1].animate.set_color(match_color),
            local_ids[1].animate.set_color(match_color),
            recon_box.animate.set_stroke(match_color, width=6),
            local_log_box.animate.set_stroke(match_color, width=6)
        )
        self.play(Indicate(eph_ids[1]), Indicate(local_ids[1]))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(alert_color)
        
        alert_rect = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.4, color=alert_color, fill_opacity=0.9)
        alert_msg = Text("POSSIBLE EXPOSURE", font_size=12, color=WHITE, weight=BOLD)
        alert = VGroup(alert_rect, alert_msg)
        self.place_at_grid(alert, "D3", scale_factor=1.1)
        
        self.play(FadeIn(alert, scale=1.2))
        self.play(Flash(alert, color=alert_color))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(local_process_color)
        
        # Border glow around the phone area
        local_process_border = Rectangle(
            height=4.5, width=3.0, 
            color=local_process_color, 
            stroke_width=2
        ).set_style(stroke_opacity=0.8)
        self.place_in_area(local_process_border, "C2", "F4")
        
        local_process_label = Text("LOCAL PROCESS", font_size=20, color=local_process_color)
        local_process_label.next_to(local_process_border, DOWN, buff=0.2)
        
        # Dim server to show lack of involvement
        self.play(
            Create(local_process_border),
            Write(local_process_label),
            server.animate.set_opacity(0.3)
        )
        self.play(local_process_border.animate.set_stroke(width=8), run_time=0.5, rate_func=there_and_back)
        self.wait(2)
