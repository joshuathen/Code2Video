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
        # Initialize the layout
        lecture_lines = [
            "DP-3T keeps data local and identities anonymous.",
            "The server never learns location or social graphs.",
            "Privacy and public health can coexist through decentralization."
        ]
        self.setup_layout("Summary & Privacy Benefits", lecture_lines)

        # Assets
        PHONE_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg"

        # Colors
        CENTRAL_COLOR = RED_A
        DP3T_COLOR = BLUE_A
        SHIELD_COLOR = "#2ecc71"
        TEXT_COLOR = WHITE

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)

        # Centralized Setup
        central_label = Text("Centralized", font_size=20, color=CENTRAL_COLOR)
        self.place_at_grid(central_label, "A1")
        
        server_cent = VGroup(
            Circle(radius=0.4, color=CENTRAL_COLOR),
            Text("Server", font_size=16, color=TEXT_COLOR)
        )
        self.place_at_grid(server_cent, "B2")
        
        # Phones for Centralized
        phone_c1 = SVGMobject(PHONE_ASSET, color=WHITE)
        phone_c2 = SVGMobject(PHONE_ASSET, color=WHITE)
        self.place_at_grid(phone_c1, "C1", scale_factor=0.5)
        self.place_at_grid(phone_c2, "C3", scale_factor=0.5)
        
        line_c1 = Line(phone_c1.get_top(), server_cent.get_bottom(), color=CENTRAL_COLOR)
        line_c2 = Line(phone_c2.get_top(), server_cent.get_bottom(), color=CENTRAL_COLOR)
        social_graph_line = Line(phone_c1.get_right(), phone_c2.get_left(), color=RED).set_stroke(opacity=0.5)
        graph_label = Text("Graph visible", font_size=12, color=RED).next_to(social_graph_line, UP, buff=0.1)
        
        central_group = VGroup(central_label, server_cent, phone_c1, phone_c2, line_c1, line_c2, social_graph_line, graph_label)

        # DP-3T Setup
        dp3t_label = Text("DP-3T", font_size=20, color=DP3T_COLOR)
        self.place_at_grid(dp3t_label, "D2") # Issue 50 fix
        
        server_dp = VGroup(
            Circle(radius=0.4, color=DP3T_COLOR),
            Text("Server", font_size=16, color=TEXT_COLOR)
        )
        self.place_at_grid(server_dp, "E2")
        
        # Phones for DP-3T (Issue 37)
        phone_d1 = SVGMobject(PHONE_ASSET, color=WHITE)
        phone_d2 = SVGMobject(PHONE_ASSET, color=WHITE)
        self.place_at_grid(phone_d1, "F1", scale_factor=0.5)
        self.place_at_grid(phone_d2, "F3", scale_factor=0.5)
        
        # Local interaction (Issue 37: lines only between phones for DP-3T)
        local_link = DoubleArrow(phone_d1.get_right(), phone_d2.get_left(), buff=0.1, color=DP3T_COLOR, tip_length=0.1)
        local_text = Text("Local Exchange", font_size=12, color=DP3T_COLOR).next_to(local_link, DOWN, buff=0.1)
        
        dp3t_group = VGroup(dp3t_label, server_dp, phone_d1, phone_d2, local_link, local_text)

        self.play(FadeIn(central_group), FadeIn(dp3t_group))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Highlight DP-3T Server as Bulletin Board
        self.play(FadeOut(central_group))
        self.play(dp3t_group.animate.scale(1.2).move_to(self.grid["C3"]))
        
        bulletin_label = Text("Bulletin Board", font_size=24, color=YELLOW)
        self.place_in_area(bulletin_label, "A2", "A5", scale_factor=0.8) # Issue 51 fix
        
        # Show "Random Keys" instead of "Identities"
        key1 = Text("0x1A2B...", font_size=14, color=DP3T_COLOR)
        key2 = Text("0x9F8E...", font_size=14, color=DP3T_COLOR)
        self.place_at_grid(key1, "D3")
        self.place_at_grid(key2, "D4")
        
        no_link_rect = Rectangle(height=1, width=3, color=RED).move_to(self.grid["E3"])
        no_link_text = Text("No Social Graph", font_size=18, color=RED).move_to(no_link_rect.get_center())

        self.play(Write(bulletin_label))
        self.play(FadeIn(key1), FadeIn(key2))
        self.play(Create(no_link_rect), Write(no_link_text))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        self.play(FadeOut(dp3t_group), FadeOut(bulletin_label), FadeOut(key1), FadeOut(key2), FadeOut(no_link_rect), FadeOut(no_link_text))

        # Final visual: Icons within Privacy Shield
        shield_rect = RoundedRectangle(corner_radius=0.3, height=4.5, width=5.5, color=SHIELD_COLOR, stroke_width=4)
        self.place_in_area(shield_rect, "A1", "F6")
        shield_label = Text("PRIVACY PRESERVED", font_size=28, color=SHIELD_COLOR, weight=BOLD)
        self.place_at_grid(shield_label, "A3", scale_factor=0.8)

        # Summary Icons (Issue 52 fixes)
        icon1_box = Rectangle(height=0.8, width=3.5, color=BLUE_B)
        icon1_text = Text("Key Generation", font_size=16, color=WHITE)
        icon1 = VGroup(icon1_box, icon1_text)
        self.place_in_area(icon1, 'B2', 'B5')

        icon2_box = Rectangle(height=0.8, width=3.5, color=BLUE_B)
        icon2_text = Text("Local Storage", font_size=16, color=WHITE)
        icon2 = VGroup(icon2_box, icon2_text)
        self.place_in_area(icon2, 'C2', 'C5')

        icon3_box = Rectangle(height=0.8, width=3.5, color=BLUE_B)
        icon3_text = Text("Local Matching", font_size=16, color=WHITE)
        icon3 = VGroup(icon3_box, icon3_text)
        self.place_in_area(icon3, 'D2', 'D5')

        self.play(Create(shield_rect), Write(shield_label))
        self.play(
            FadeIn(icon1, shift=UP),
            FadeIn(icon2, shift=UP),
            FadeIn(icon3, shift=UP)
        )
        self.wait(3)

        # Cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(1)
