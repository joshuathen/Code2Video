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
        # Initialization
        title = "Step 3: The Diagnosis Report (Sharing the Key, Not the Contacts)"
        lines = [
            "If Alice tests positive, she uploads her Secret Key.",
            "She never shares her meeting history or location data.",
            "The server broadcasts infected keys to all system users."
        ]
        self.setup_layout(title, lines)

        # Assets Creation
        # Alice's Phone [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg]
        alice_phone = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg", color=BLUE)
        alice_label = Text("Alice", font_size=16, color=BLUE)
        alice_group = VGroup(alice_phone, alice_label).arrange(DOWN, buff=0.2)
        self.place_at_grid(alice_group, "B2", scale_factor=0.6)

        # Server
        server_rect = Rectangle(height=1.2, width=1.5, color=GREY_A)
        server_label = Text("Server", font_size=16, color=GREY_A)
        server_group = VGroup(server_rect, server_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(server_group, "B5", scale_factor=1.0)

        # Bob's Phone [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg]
        bob_phone = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg", color=GREEN)
        bob_label = Text("Bob", font_size=16, color=GREEN)
        bob_group = VGroup(bob_phone, bob_label).arrange(DOWN, buff=0.2)
        # Fix Issue 41: scaling and position
        self.place_at_grid(bob_group, "E2", scale_factor=0.85)

        # Infected Keys List
        list_rect = Rectangle(height=2.0, width=1.5, color=WHITE)
        list_title = Text("Infected Keys", font_size=18, color=WHITE)
        list_title.move_to(list_rect.get_top() + DOWN * 0.4)
        list_group = VGroup(list_rect, list_title)
        # Fix Issue 40: internal overlap and scaling
        self.place_in_area(list_group, "E5", "F6", scale_factor=0.6)

        # Secret Key SK_t
        sk_t = Text("SK_t", color="#FFFF00", font_size=20)
        
        # Privacy Protection Icons
        location_icon = VGroup(Circle(radius=0.2, color=RED), Line(UP, DOWN, color=RED), Line(LEFT, RIGHT, color=RED)).rotate(PI/4)
        history_icon = VGroup(Square(side_length=0.4, color=RED), Line(UP, DOWN, color=RED), Line(LEFT, RIGHT, color=RED)).rotate(PI/4)
        privacy_icons = VGroup(location_icon, history_icon).arrange(RIGHT, buff=0.3)
        # Fix Issue 39: Scale and place to avoid overlap with Alice label
        self.place_at_grid(privacy_icons, "C2", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(alice_group), Create(server_group))
        
        sk_t.move_to(alice_group.get_center())
        self.play(FadeIn(sk_t))
        # Alice sends SK_t to server
        self.play(sk_t.animate.move_to(server_group.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Privacy demonstration: Not sending meeting history or location
        self.play(FadeIn(privacy_icons))
        cross_line = Line(privacy_icons.get_left(), privacy_icons.get_right(), color=RED, stroke_width=4)
        self.play(Create(cross_line))
        
        # Server updates list with the new key
        self.play(Create(list_group))
        sk_t_in_list = sk_t.copy().scale(0.8)
        # Place key inside the visual list box
        sk_t_in_list.move_to(list_group[0].get_center() + DOWN * 0.2)
        self.play(sk_t_in_list.animate.move_to(list_group[0].get_center() + DOWN * 0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(Create(bob_group))
        
        # Bob downloads updated list
        # Visualizing the download as a copy of the list+key moving to Bob's phone
        download_packet = VGroup(list_group[0].copy(), sk_t_in_list.copy())
        self.play(download_packet.animate.move_to(bob_group.get_center()).scale(0.4))
        self.play(FadeOut(download_packet))
        
        self.wait(2)
