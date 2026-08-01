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
        # Data
        title = "The Local Match: Privacy-First Verification"
        lecture_lines = [
            "Phones download new Secret Day Keys from the server.",
            "Your device reconstructs the infected person's rotating IDs.",
            "It checks these IDs against its own local diary.",
            "A match triggers a notification on your device.",
            "The matching process happens entirely on your phone."
        ]
        
        # Colors
        COLOR_KEYS = "#FFD700"
        COLOR_RECON = "#87CEEB"
        COLOR_DIARY = "#F5DEB3"
        COLOR_MATCH = "#FFFF00"
        
        self.setup_layout(title, lecture_lines)
        
        # Assets
        SERVER_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/server.svg"
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        # Alice (Phone) and Server setup using SVGMobjects
        alice_phone = SVGMobject(PHONE_ASSET).set_color(WHITE)
        alice_label = Text("Alice's Phone", font_size=16).next_to(alice_phone, DOWN, buff=0.1)
        alice_group = VGroup(alice_phone, alice_label)
        self.place_at_grid(alice_group, "A2", scale_factor=0.6)
        
        server_icon = SVGMobject(SERVER_ASSET).set_color(GRAY)
        server_label = Text("Server", font_size=16).next_to(server_icon, DOWN, buff=0.1)
        server_group = VGroup(server_icon, server_label)
        self.place_at_grid(server_group, "A6", scale_factor=0.7)
        
        key_icon = Star(n=5, color=COLOR_KEYS).scale(0.2)
        key_label = Text("Secret Key", font_size=14, color=COLOR_KEYS).next_to(key_icon, UP, buff=0.1)
        key_group = VGroup(key_icon, key_label)
        # Issue 40: Scale down key_group to 0.5 to avoid crowding
        self.place_at_grid(key_group, "A6", scale_factor=0.5)
        
        self.play(FadeIn(alice_group), FadeIn(server_group))
        self.play(self.lecture[0].animate.set_color(COLOR_KEYS))
        self.play(key_group.animate.move_to(self.grid["A2"]))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Move phone group to row C to start processing
        self.play(alice_group.animate.move_to(self.grid["C2"]), key_group.animate.move_to(self.grid["C2"]))
        
        hash_box = Rectangle(width=1.2, height=0.6, color=WHITE).set_fill(BLUE_E, opacity=0.3)
        hash_text = Text("Hash Function", font_size=14)
        hash_group = VGroup(hash_box, hash_text)
        self.place_at_grid(hash_group, "C4")
        
        self.play(self.lecture[1].animate.set_color(COLOR_RECON))
        self.play(FadeIn(hash_group))
        
        recon_ids_label = Text("Reconstructed IDs", font_size=16, color=COLOR_RECON)
        self.place_at_grid(recon_ids_label, "D2")
        
        id_list = VGroup(
            Text("8F3D... (Recon)", font_size=14, color=COLOR_RECON),
            Text("2A7C... (Recon)", font_size=14, color=COLOR_RECON),
            Text("5B9E... (Recon)", font_size=14, color=COLOR_RECON)
        ).arrange(DOWN, buff=0.2)
        # Issue 41: Use area positioning for lists to prevent clipping
        self.place_in_area(id_list, "D4", "D6", scale_factor=0.8)
        
        arrow_to_hash = Arrow(start=self.grid["C2"], end=self.grid["C4"], buff=0.4, color=WHITE)
        # Center of D4-D6 area is D5
        arrow_to_ids = Arrow(start=self.grid["C4"], end=self.grid["D5"], buff=0.4, color=WHITE)
        
        self.play(Create(arrow_to_hash))
        self.play(Create(arrow_to_ids), Write(recon_ids_label))
        self.play(Create(id_list))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        diary_label = Text("Local Diary", font_size=16, color=COLOR_DIARY)
        self.place_at_grid(diary_label, "E2")
        
        diary_list = VGroup(
            Text("1C2B... (Diary)", font_size=14, color=COLOR_DIARY),
            Text("8F3D... (Diary)", font_size=14, color=COLOR_DIARY),
            Text("7D4F... (Diary)", font_size=14, color=COLOR_DIARY)
        ).arrange(DOWN, buff=0.2)
        # Issue 41: Use area positioning for local diary list
        self.place_in_area(diary_list, "E4", "E6", scale_factor=0.8)
        
        self.play(self.lecture[2].animate.set_color(COLOR_DIARY))
        self.play(Write(diary_label), Create(diary_list))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_MATCH))
        
        # Highlight matching "8F3D" entry in both lists
        match_highlight_recon = SurroundingRectangle(id_list[0], color=COLOR_MATCH)
        match_highlight_diary = SurroundingRectangle(diary_list[1], color=COLOR_MATCH)
        
        self.play(Create(match_highlight_recon), Create(match_highlight_diary))
        self.play(id_list[0].animate.set_color(COLOR_MATCH), diary_list[1].animate.set_color(COLOR_MATCH))
        
        risk_alert = Text("!!! RISK ALERT !!!", font_size=20, color=RED_A).set_background_stroke(color=WHITE, width=1)
        # Issue 42: Use area positioning for multi-token label
        self.place_in_area(risk_alert, "B1", "B3", scale_factor=0.8)
        
        self.play(Flash(risk_alert, color=YELLOW), Write(risk_alert))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        local_frame = DashedVMobject(Rectangle(width=5.5, height=4.5, color=GREEN_B))
        self.place_in_area(local_frame, "C1", "F6")
        local_tag = Text("ON-DEVICE PROCESSING", font_size=14, color=GREEN_B).next_to(local_frame, DOWN, buff=0.1)
        
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.play(Create(local_frame), FadeIn(local_tag))
        
        # Emphasize privacy by fading the server
        self.play(server_group.animate.set_opacity(0.2))
        self.wait(2)
