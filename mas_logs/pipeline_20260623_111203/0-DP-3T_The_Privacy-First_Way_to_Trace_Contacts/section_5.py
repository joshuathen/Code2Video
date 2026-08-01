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
        self.setup_layout("Step 3: Positive Diagnosis and Reporting", 
                          ['If Alice tests positive, she voluntarily reports her keys.', 
                           'She uploads her Daily Secret Keys to a server.', 
                           'Crucially, no personal identities or locations are shared.', 
                           'The server only sees a list of anonymous keys.', 
                           'This protects Alice’s privacy while enabling notification.'])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#E74C3C")
        
        alice_phone = RoundedRectangle(height=1.4, width=0.8, color=WHITE, corner_radius=0.1)
        self.place_at_grid(alice_phone, "A2")
        
        positive_icon = Cross(stroke_width=6, scale_factor=0.2).set_color(WHITE)
        self.place_at_grid(positive_icon, "A2")
        
        status_label = Text("POSITIVE", color=WHITE, font_size=16)
        self.place_at_grid(status_label, "B2")
        
        prompt_box = RoundedRectangle(height=0.4, width=1.5, color=BLUE, fill_opacity=0.2)
        prompt_text = Text("Upload Keys?", color=BLUE, font_size=14)
        prompt_vgroup = VGroup(prompt_box, prompt_text)
        self.place_at_grid(prompt_vgroup, "A4")

        self.play(Create(alice_phone), Write(status_label))
        self.play(
            alice_phone.animate.set_fill("#E74C3C", opacity=0.8),
            FadeIn(positive_icon)
        )
        self.play(FadeIn(prompt_vgroup))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#95A5A6")
        
        server = Square(side_length=1.4, color="#95A5A6", fill_opacity=0.9)
        server_label = Text("Diagnosis\nServer", color=WHITE, font_size=18)
        self.place_at_grid(server, "E5")
        server_label.move_to(server.get_center())
        
        # Keys to upload
        sk_keys = VGroup(*[
            Rectangle(height=0.15, width=0.4, color=WHITE, fill_opacity=1) 
            for _ in range(3)
        ]).arrange(DOWN, buff=0.1)
        self.place_at_grid(sk_keys, "A2")
        
        upload_arrow = Arrow(self.grid["A4"], self.grid["D5"], color=WHITE, buff=0.2)

        self.play(FadeIn(server), FadeIn(server_label))
        self.play(Create(upload_arrow))
        
        # Animate keys moving to server
        self.play(
            sk_keys.animate.move_to(self.grid["E5"]).scale(0.5).set_opacity(0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#2ECC71")
        
        # Personal Data Icons
        id_rect = Rectangle(height=0.3, width=0.5, color=WHITE)
        id_line = Line(LEFT*0.15, RIGHT*0.15, color=WHITE).shift(UP*0.05)
        id_icon = VGroup(id_rect, id_line)
        
        loc_pin_circle = Circle(radius=0.1, color=WHITE)
        loc_pin_base = Triangle(color=WHITE).scale(0.15).rotate(PI).shift(DOWN*0.1)
        loc_icon = VGroup(loc_pin_circle, loc_pin_base)
        
        self.place_at_grid(id_icon, "C2")
        self.place_at_grid(loc_icon, "D2")
        
        shield = Circle(radius=0.5, color="#2ECC71", fill_opacity=0.4)
        shield_label = Text("Privacy Shield", color="#2ECC71", font_size=12)
        self.place_at_grid(shield, "C3")
        shield_label.next_to(shield, DOWN, buff=0.1)
        
        self.play(FadeIn(id_icon), FadeIn(loc_icon), Write(shield_label))
        self.play(GrowFromCenter(shield))
        
        # Block movement
        self.play(
            id_icon.animate.shift(RIGHT*0.6),
            loc_icon.animate.shift(RIGHT*0.6)
        )
        self.play(Indicate(shield, color=RED, scale_factor=1.1))
        self.play(
            id_icon.animate.shift(LEFT*0.6),
            loc_icon.animate.shift(LEFT*0.6)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#ECF0F1")
        
        hex_list = VGroup(
            Text("0x4E1A9...", font_size=14, color="#ECF0F1"),
            Text("0xF2B91...", font_size=14, color="#ECF0F1"),
            Text("0x7C3D5...", font_size=14, color="#ECF0F1")
        ).arrange(DOWN, buff=0.1)
        self.place_at_grid(hex_list, "E4")
        
        anon_tag = Text("Anonymous", color=WHITE, font_size=12).next_to(hex_list, UP, buff=0.1)
        
        self.play(FadeIn(anon_tag), Write(hex_list))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#F1C40F")
        
        notif_circle = Circle(radius=0.3, color="#F1C40F", fill_opacity=0.5)
        notif_icon = Star(n=5, color="#F1C40F", fill_opacity=1).scale(0.2)
        notif_group = VGroup(notif_circle, notif_icon)
        self.place_at_grid(notif_group, "E6")
        
        # Broadness arrows
        broadcast_arrows = VGroup(*[
            Arrow(self.grid["E5"], self.grid[pos], color="#F1C40F", buff=0.4)
            for pos in ["D6", "F4", "F6"]
        ])

        self.play(FadeIn(notif_group))
        self.play(
            notif_group.animate.scale(1.3),
            Create(broadcast_arrows),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
